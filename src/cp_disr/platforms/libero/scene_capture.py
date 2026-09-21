"""Read-only deterministic MuJoCo RGB/depth capture for Stage 0C inputs."""
from pathlib import Path
import hashlib,json
import numpy as np
import mujoco

class SceneCaptureAdapter:
    def __init__(self, xml_template, width=128, height=128):
        self.xml_template=Path(xml_template); self.width=width; self.height=height
        self.model=None; self.data=None; self.renderer=None; self.config=None
    @staticmethod
    def build_xml(config):
        def box(name,pos,size,color):
            p=' '.join(f'{float(x):.6f}' for x in pos); s=' '.join(f'{float(x):.6f}' for x in size); c=' '.join(str(float(x)) for x in color)
            return f'<body name="{name}" pos="{p}"><geom type="box" size="{s}" rgba="{c}"/></body>'
        obj=config['objects']; c=config['container']; b=config['buffer']
        return f'''<mujoco model="cp_disr_{config['task_id']}"><compiler angle="radian"/><option timestep="0.01" gravity="0 0 -9.81"/>
<worldbody><light pos="0 -2 4" dir="0 0 -1" diffuse="1 1 1"/>
<geom name="floor" type="plane" size="3 3 .01" rgba=".70 .72 .75 1"/>
<geom name="table" type="box" pos="0 0 0.03" size="1.45 1.05 .03" rgba=".48 .30 .16 1"/>
<camera name="agentview" pos="0 -3.4 2.8" xyaxes="1 0 0 0 .63 .78"/>
<geom name="buffer" type="box" pos="{b[0]} {b[1]} 0.095" size="{b[2]} {b[3]} .012" rgba=".18 .72 .30 1"/>
<geom name="container_base" type="box" pos="{c[0]} {c[1]} 0.12" size=".30 .24 .025" rgba=".18 .35 .75 1"/>
<geom name="container_back" type="box" pos="{c[0]} {c[1]+.215} 0.25" size=".30 .025 .15" rgba=".18 .35 .75 1"/>
<geom name="container_left" type="box" pos="{c[0]-.275} {c[1]} 0.25" size=".025 .24 .15" rgba=".18 .35 .75 1"/>
<geom name="container_right" type="box" pos="{c[0]+.275} {c[1]} 0.25" size=".025 .24 .15" rgba=".18 .35 .75 1"/>
{box('target',obj['target'],(.10,.10,.10),(.85,.20,.15,1))}
{box('second_object',obj.get('second_object',obj.get('interferer')), (.10,.10,.10),(.92,.72,.12,1))}
</worldbody><actuator/></mujoco>'''
    def reset_scene(self, task_ref, scene_config, seed):
        self.config=dict(scene_config); self.config['seed']=int(seed)
        xml=self.build_xml(self.config); self.model=mujoco.MjModel.from_xml_string(xml); self.data=mujoco.MjData(self.model); mujoco.mj_forward(self.model,self.data)
        self.renderer=mujoco.Renderer(self.model,height=self.height,width=self.width); self.renderer.update_scene(self.data,camera='agentview')
    def capture_rgb(self):
        self.renderer.update_scene(self.data,camera='agentview'); return np.asarray(self.renderer.render()).copy()
    def capture_depth(self):
        self.renderer.enable_depth_rendering(); self.renderer.update_scene(self.data,camera='agentview'); d=np.asarray(self.renderer.render()).copy(); self.renderer.disable_depth_rendering(); return d
    def public_scene_metadata(self):
        return {'task_id':self.config['task_id'],'seed':self.config['seed'],'camera':'agentview','width':self.width,'height':self.height,'platform_type':'simulator','hidden_truth_used_for_prompt':False}
    def close(self):
        if self.renderer is not None: self.renderer.close()
        self.renderer=self.model=self.data=None
