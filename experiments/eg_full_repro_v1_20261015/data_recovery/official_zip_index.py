"""Get independent official ZIP directory CRCs through anonymous HTTP ranges.

No dataset images or full ZIPs are downloaded on this computer. Model execution
and all actual dataset preparation remain on lab-xushijie3.
"""
import concurrent.futures
import hashlib
import io
import json
import re
import time
import urllib.parse
import urllib.error
import urllib.request
import zipfile
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath


class Form(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action, self.inputs = "", {}

    def handle_starttag(self, tag, attrs):
        fields = dict(attrs)
        if tag == "form":
            self.action = fields.get("action", "")
        elif tag == "input" and "name" in fields:
            self.inputs[fields["name"]] = fields.get("value", "")


class RemoteZip(io.RawIOBase):
    def __init__(self, url):
        super().__init__()
        self.url, self.position, self.network_bytes = url, 0, 0
        body, content_range = self.request(0, 0)
        assert len(body) == 1 and content_range.startswith("bytes 0-0/")
        self.size = int(content_range.split("/")[-1])

    def request(self, start, end):
        for attempt in range(3):
            req = urllib.request.Request(self.url + "&egnonce=" + str(time.time_ns()), headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity", "Cache-Control": "no-cache"})
            with urllib.request.urlopen(req, timeout=60) as reply:
                if reply.status != 206:
                    problem = f"HTTP {reply.status} type={reply.headers.get('Content-Type')} range={start}-{end}"
                    if attempt < 2:
                        time.sleep(2)
                        continue
                    raise RuntimeError(problem)
                content_range = reply.headers.get("Content-Range", "")
                assert content_range.startswith(f"bytes {start}-{end}/"), content_range
                body = reply.read(end - start + 2)
            break
        assert len(body) == end - start + 1
        self.network_bytes += len(body)
        return body, content_range

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        position = offset if whence == io.SEEK_SET else self.position + offset if whence == io.SEEK_CUR else self.size + offset
        if position < 0:
            raise ValueError("negative seek")
        self.position = position
        return position

    def read(self, count=-1):
        count = self.size - self.position if count < 0 else min(count, self.size - self.position)
        if count <= 0:
            return b""
        assert count <= 128 * 1024**2, "unexpected oversized ZIP-directory read"
        data, _ = self.request(self.position, self.position + count - 1)
        self.position += len(data)
        return data


def group_for(name):
    parts = PurePosixPath(name).parts
    if parts and parts[0] == "graspnet":
        parts = parts[1:]
    if parts and re.fullmatch(r"scene_\d{4}", parts[0]):
        parts = ("scenes",) + parts
    if not parts or ".." in parts:
        return None
    name = "/".join(parts)
    if parts[0] == "scenes" and len(parts) >= 3:
        if not re.fullmatch(r"scene_\d{4}", parts[1]) or not 0 <= int(parts[1][6:]) < 190:
            return None
        if (len(parts) == 3 and parts[2] in {"object_id_list.txt", "rs_wrt_kn.npy"}) or (
            parts[2] == "kinect" and (
                len(parts) == 4 and parts[3] in {"camK.npy", "camera_poses.npy", "cam0_wrt_table.npy"}
                or len(parts) == 5 and parts[3] in {"rgb", "depth", "label", "meta", "annotations"}
            )
        ):
            return parts[1], name
    elif parts[0] == "grasp_label" and name.endswith(".npz"):
        return "grasp_label", name
    elif parts[0] == "collision_label" and len(parts) >= 3 and parts[1].startswith("scene_") and int(parts[1][6:]) < 100:
        return "/".join(parts[:2]), name
    elif parts[0] == "models" and len(parts) >= 3 and parts[1].isdigit() and 0 <= int(parts[1]) < 88:
        return "/".join(parts[:2]), name
    return None


def _index_one(source):
    cache = Path(__file__).parent / "official_zip_indexes" / (source["archive"] + ".json")
    cache.parent.mkdir(exist_ok=True)
    if cache.exists():
        return json.loads(cache.read_text())
    file_id = re.search(r"/d/([^/]+)", source["google"]).group(1)
    with urllib.request.urlopen("https://drive.google.com/uc?export=download&id=" + file_id, timeout=30) as reply:
        html = reply.read(1000000).decode()
    form = Form()
    form.feed(html)
    assert form.action.startswith("https://drive.usercontent.google.com/download")
    assert form.inputs.get("id") == file_id
    stream = RemoteZip(form.action + "?" + urllib.parse.urlencode(form.inputs))
    records = []
    raw_names = []
    with zipfile.ZipFile(stream) as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            if len(raw_names) < 5:
                raw_names.append(entry.filename)
            match = group_for(entry.filename)
            if match:
                group, name = match
                records.append({"group": group, "path": name, "bytes": entry.file_size, "crc32": f"{entry.CRC:08x}"})
    print(json.dumps({"archive": source["archive"], "status": "INDEXED", "selected_files": len(records), "network_bytes": stream.network_bytes, "sample_names": raw_names}), flush=True)
    assert records, f"No canonical dataset paths found: {raw_names}"
    result = {"archive": source["archive"], "official_url": source["google"], "zip_bytes": stream.size,
              "network_bytes": stream.network_bytes, "records": records}
    cache.write_text(json.dumps(result))
    return result


def index_one(source):
    for attempt in range(3):
        try:
            return _index_one(source)
        except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
            print(json.dumps({"archive": source["archive"], "retry": attempt + 1, "error": str(exc)}), flush=True)
            if attempt == 2:
                raise
            time.sleep(3)


if __name__ == "__main__":
    root = Path(__file__).parent
    sources = json.loads((root / "OFFICIAL_DATA_SOURCES.json").read_text(encoding="utf-8-sig"))["sources"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        indexes = list(pool.map(index_one, sources))
    groups, seen = {}, {}
    for index in indexes:
        for record in index["records"]:
            if record["path"] in seen:
                assert seen[record["path"]] == record, "conflicting official ZIP entries"
                continue
            seen[record["path"]] = record
            groups.setdefault(record["group"], []).append(record)
    summaries = {}
    for group, records in groups.items():
        canonical = "".join(f"{r['path']}\t{r['bytes']}\t{r['crc32']}\n" for r in sorted(records, key=lambda r: r["path"]))
        summaries[group] = {"files": len(records), "bytes": sum(r["bytes"] for r in records),
                            "sha256_of_path_size_crc32_list": hashlib.sha256(canonical.encode()).hexdigest()}
    assert all(f"scene_{i:04d}" in summaries for i in range(190)), "missing official scene groups"
    assert "grasp_label" in summaries and all(f"models/{i:03d}" in summaries for i in range(88))
    summary = {"source": "Official GraspNet Google Drive ZIP central directories", "anonymous": True,
               "method": "HTTP Range; paths, uncompressed sizes and CRC32; full ZIPs not downloaded",
               "groups": summaries, "archives": [{k: v for k, v in x.items() if k != "records"} for x in indexes]}
    (root / "official_zip_crc_summary.json").write_text(json.dumps(summary, indent=2))
    (root / "official_zip_crc_records.json").write_text(json.dumps(indexes))
    print(json.dumps({"status": "COMPLETE", "groups": len(summaries), "files": len(seen)}), flush=True)
