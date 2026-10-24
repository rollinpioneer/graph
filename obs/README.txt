【未完成：cal_0065_0000 的 I1 回放误差触发停止；只导出 25/144 帧】
详见 partial_report.txt 和 summary.json。以下为原定流程，不代表全部执行完成。

观测接触几何导出（2026-10-24）

计划仅 cal、C3、P2 共 144 帧，直接使用 v3/eval 的既有 pred，不生成或筛选新姿态。
不读取这些文件中的摩擦、碰撞标签；没有推理、训练、换角选择或相关性分析。
主观测分支只用整幅有效深度：Open3D 统计离群点去除(20,2.0)，
在去飞点后的点云上建立 KD 树，1 cm 半径、至少 10 点的 PCA 法向朝向相机。
2/3/5 mm 管道和接触跨度阈值严格按 export_instruction.md。
相同极值时取点索引靠前者。单侧不足 5 mm 时用最小 y 点法向判侧，
负向取最小 y 点为侧 0，否则取最大 y 点为侧 1；此实现约定写入 protocol.json。
分割仅用于单独的 seg_r3 输出；其法向仍来自整幅去飞点后的点云。
像素标签 = 官方零基 object_id + 1，0 为背景，每帧与 meta cls_indexes 核验。

GT 接触采用官方 collision_detection 中相同的 GraspGroup→Dex-Net 构造，
仅对 v3 I1 有值的姿态调用 close_fingers(check_approach=False)。
未运行 collision_detection 或 get_grasp_score，未计算相关统计。
用官方 get_model_poses 的物体到相机变换保存 gt_pts_cam。
I1 在与 v3 相同的物体坐标系下逐字使用原公式复算。
原先缺失的 I1 位置按指令保持缺失，不声称独立复核了这些缺失样本。

先 C3_0100_0000 全字段两次字节级比较，再 cal_0067_0000 核验；
两帧通过后才执行剩余帧。每帧检查 GT 闭合线/顺序/宽度联合比例≥99%、
I1 缺失位置相同且最大误差<1e-7、候选 ID/原角度一致；失败立即停止。
CPU 环境与 v3 相同，禁用 CUDA，BLAS/OMP/MKL 单线程，共 12 个工作进程。
在激活环境后进入本目录：python export_obs.py pilot；通过后 python export_obs.py 12；
完成后 python summarize.py。依赖在相同工作根目录下的既有官方评价代码和数据。

每帧 NPZ 与 JSON 收据保存相应数据和 SHA。summary.json 只给指定的
计数、去飞点比例、found 比例、四项核验、标签对应和 CPU 耗时。
output_manifest.json 用于检查下载和 Git 归档的完整性。
