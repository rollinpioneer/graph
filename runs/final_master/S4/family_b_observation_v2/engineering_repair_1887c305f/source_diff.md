# Source diff vs base 1887c305f (binding repair only)

```diff
diff --git a/src/cp_disr/platforms/libero/family_b_obs_v2.py b/src/cp_disr/platforms/libero/family_b_obs_v2.py
index 76f03f01a..614b1c4df 100644
--- a/src/cp_disr/platforms/libero/family_b_obs_v2.py
+++ b/src/cp_disr/platforms/libero/family_b_obs_v2.py
@@ -102,11 +102,62 @@ def calibration(env, name):
     }
 
 
-def verify_frozen_cameras(env, frozen):
+FROZEN_PROFILE_SHA256 = "3c98a09649a73a3d9cee1db7e7a9f67063e515bfe5c3e689cabf4cf09900d917"
+CAMERA_REF_FIELDS = ("name", "mode", "pos", "quat", "fovy", "width", "height", "fixed_in_world")
+
+
+class FrozenCameraProfileError(ValueError):
+    """The loaded observation profile does not have the frozen structure/identity."""
+
+
+def selected_camera_refs(observation_profile):
+    """Validate the FULL observation_profile_v2 structure; return {camera: frozen ref}.
+
+    Structure of the loaded profile (not of the standalone camera_manifest.json):
+      profile["cameras"]         -> list of camera names
+      profile["camera_manifest"] -> dict keyed by camera name -> frozen camera parameters
+    Fail closed: no fallback to the arena catalogue, no update from live values.
+    """
+    prof = observation_profile
+    if not isinstance(prof, dict):
+        raise FrozenCameraProfileError("profile must be a dict")
+    if prof.get("profile_version") != PROFILE_VERSION:
+        raise FrozenCameraProfileError("profile_version")
+    if "cameras" not in prof or "camera_manifest" not in prof:
+        raise FrozenCameraProfileError("profile lacks cameras/camera_manifest")
+    camera_names = prof["cameras"]
+    camera_refs = prof["camera_manifest"]
+    if not isinstance(camera_names, (list, tuple)):
+        raise FrozenCameraProfileError("profile['cameras'] must be a list/tuple of names")
+    if not isinstance(camera_refs, dict):
+        raise FrozenCameraProfileError("profile['camera_manifest'] must be a dict keyed by camera name")
+    if tuple(camera_names) != CAMERAS:
+        raise FrozenCameraProfileError("camera names differ from the frozen camera set")
+    if set(camera_names) != set(camera_refs.keys()):
+        raise FrozenCameraProfileError("camera list and camera_manifest keys differ")
+    for name in camera_names:
+        ref = camera_refs[name]
+        if not isinstance(ref, dict):
+            raise FrozenCameraProfileError(f"{name}: camera ref must be a dict")
+        missing = [k for k in CAMERA_REF_FIELDS if k not in ref]
+        if missing:
+            raise FrozenCameraProfileError(f"{name}: missing fields {missing}")
+        if ref["name"] != name or ref["fixed_in_world"] is not True:
+            raise FrozenCameraProfileError(f"{name}: name/fixed_in_world")
+        if int(ref["width"]) != IMAGE_SIZE or int(ref["height"]) != IMAGE_SIZE:
+            raise FrozenCameraProfileError(f"{name}: resolution")
+    body = {k: v for k, v in prof.items() if k != "profile_sha256"}
+    if prof.get("profile_sha256") != FROZEN_PROFILE_SHA256 or digest(body) != FROZEN_PROFILE_SHA256:
+        raise FrozenCameraProfileError("profile_sha256 differs from the frozen value")
+    return {name: camera_refs[name] for name in camera_names}
+
+
+def verify_frozen_cameras(env, observation_profile):
     """Live fixed cameras must match the frozen manifest exactly (before reset)."""
+    camera_refs = selected_camera_refs(observation_profile)
     issues = []
     for name in CAMERAS:
-        live, ref = live_camera(env, name), frozen["cameras"][name]
+        live, ref = live_camera(env, name), camera_refs[name]
         if live["mode"] != ref["mode"]:
             issues.append(f"{name}:mode")
         for key in ("pos", "quat"):
@@ -143,8 +194,7 @@ class FamilyBObsV2Env(FamilyBEnv):
             camera_widths=[IMAGE_SIZE] * len(CAMERAS),
             camera_depths=[True] * len(CAMERAS), camera_segmentations=None,
         )
-        if _PROFILE["data"] is not None:
-            verify_frozen_cameras(self, _PROFILE["data"])
+        verify_frozen_cameras(self, profile())  # fail closed: profile must be loaded
 
     def public_observation(self):
         """Same public fields as D0 (agentview kept as rgb/depth) plus all fixed views."""
```

Tests: tests/test_family_b_obs_v2.py gains R-FC01..R-FC10 (16 collected cases); no other source, config or result file changed.
