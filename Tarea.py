from pyglet.window import Window 
from pyglet.app import run
from pyglet import clock
from pyglet.graphics.shader import Shader, ShaderProgram
from OpenGL.GL import *
from OpenGL.GL.shaders import *

from itertools import chain
from pathlib import Path

import pyglet

import math as mt
import numpy as np

import pyglet.window as pw
from libs.gpu_shape import createGPUShape
from libs.obj_handler import read_OBJ2
from libs.obj_handler import read_OBJ

import libs.shaders as sh
import libs.transformations as tr
import libs.scene_graph as sg
import libs.curves as cv

vertex_shader = """
#version 330

uniform mat4 model;
uniform mat4 transform;
uniform mat4 view;
uniform mat4 projection;

in vec3 position;
in vec2 texCoords;

out vec2 outTexCoords;

void main()
{
    vec4 vertexPos = model * vec4(position, 1.0);
    gl_Position = projection * view * transform * vertexPos;
    outTexCoords = vec2(texCoords[0], 1 - texCoords[1]);
}
"""

fragment_shader = """
#version 330

uniform sampler2D samplerTex;

in vec2 outTexCoords;
out vec4 outColor;

void main()
{
    vec4 textureColor = texture(samplerTex, outTexCoords);
    outColor = textureColor;
}
"""

point_curve_shader = """
#version 330
uniform mat4 view;
uniform mat4 projection;

in vec3 position;

void main()
{
    gl_PointSize = 1.0;
    gl_Position = projection * view * vec4(position, 1.0);
}
"""

fragment_curve_shader = """
#version 330
out vec4 outColor;

void main()
{
    outColor = vec4(1.0, 1.0, 1.0, 1.0);
}
"""

control_point_shader = """
#version 330
uniform mat4 view;
uniform mat4 projection;

in vec3 position;

void main()
{
    gl_PointSize = 7.5;
    gl_Position = projection * view * vec4(position, 1.0);
}
"""

fragment_control_shader = """
#version 330
out vec4 outColor;

void main()
{
    outColor = vec4(0.8, 0.99, 0.31, 1.0);
}
"""

vertex_shader_particle = """
#version 330
uniform mat4 view;
uniform mat4 projection;
uniform float max_ttl;

in vec3 position;
in float ttl;
out float alpha;

void main()
{
    gl_PointSize = 15.0 * (ttl / max_ttl);
    gl_Position = projection * view * vec4(position, 1.0);
    alpha = ttl / max_ttl;
}
"""

fragment_shader_particle = """
#version 330

in float alpha;
out vec4 outColor;

void main()
{
    outColor = vec4(1.0, 1.0, 1.0, alpha);
}
"""

#Cargar los pipelines a usar
PIPELINE = sh.SimpleTextureModelViewProjectionShaderProgram(vertex_shader, fragment_shader)
PHONG_PIPELINE = sh.MultipleLightTexturePhongShaderProgram()
pipeline_particles = ShaderProgram(Shader(vertex_shader_particle, "vertex"), Shader(fragment_shader_particle, "fragment"))
pipeline_curve = ShaderProgram(Shader(point_curve_shader, "vertex"), Shader(fragment_curve_shader, "fragment"))
pipeline_control_points = ShaderProgram(Shader(control_point_shader, "vertex"), Shader(fragment_control_shader, "fragment"))

#Cargar los assets
ASSETS = { "Ship": Path("assets") / "Ship.obj", "Ship_Texture": Path("assets") / "Ship_Texture.png",
          "Shadow_Texture": Path("assets") / "Shadow_Texture.png",
           "Floor": Path("assets") / "Floor.obj", "Floor_Texture": Path("assets") / "Floor_Texture.png",
           "Sun": Path("assets") / "Sun.obj", "Sun_Texture": Path("assets") / "Sun_Texture.png", 
           "Pillar": Path("assets") / "Pillar.obj", "Pillar_Texture": Path("assets") / "Pillar_Texture.png", 
           "Deer": Path("assets") / "Deer.obj", "Deer_Texture": Path("assets") / "Deer_Texture.png", 
           "Portal_cube": Path("assets") / "Portal_cube.obj", "Portal_cube_Texture": Path("assets") / "Portal_cube_Texture.png",
           "Star": Path("assets") / "Star.obj", "Star_Textures": Path("assets") / "Star_Textures.png",
           "Star_Phong_Textures": Path("assets") / "Star_Phong_Textures.png", "Song_1": Path("assets") / "Rubble_Bobble.wav",
           "Song_2": Path("assets") / "Retrograde.wav"

}

#Clase que tiene todos los datos actuales de las naves
class Ship:
    def __init__(self, pipeline, phong_pipeline):
        #Cargando la GPU_Shape y Textura de la nave
        self.ex_shape = createGPUShape(pipeline, read_OBJ2(ASSETS["Ship"]))

        self.tex_params = [GL_REPEAT, GL_REPEAT, GL_NEAREST, GL_NEAREST]
        self.current_tex = ASSETS["Ship_Texture"]

        self.ex_shape.texture = sh.textureSimpleSetup(
            self.current_tex, *self.tex_params
        )

        self.shadow_shape = createGPUShape(pipeline, read_OBJ2(ASSETS["Ship"]))
        self.shadow_shape.texture = sh.textureSimpleSetup(ASSETS["Shadow_Texture"], *self.tex_params)

        #GPU_Shape con Pipeline con iluminación
        self.phong_shape = createGPUShape(phong_pipeline, read_OBJ2(ASSETS["Ship"]))

        self.phong_shape.texture = sh.textureSimpleSetup(
            self.current_tex, *self.tex_params
        )

        self.phong_shadow_shape = createGPUShape(phong_pipeline, read_OBJ2(ASSETS["Ship"]))
        self.phong_shadow_shape.texture = sh.textureSimpleSetup(ASSETS["Shadow_Texture"], *self.tex_params)

        #Parametros de la nave
        self.pos = np.array([0,0,0], dtype = float)
        self.front = np.array([1,0,0], dtype = float)
        self.move = np.array([0,0,0], dtype = float)
        self.up = np.array([0,0,1], dtype = float)

        #Angulos del vector front de la nave
        self.theta = mt.acos(self.front[2])
        self.phi = mt.atan(self.front[1]/self.front[0])
        
        #Parametros utiles
        self.time = 0
        self.zoom = 0
        self.fovi = 0
        self.look = 0
        self.step = 0
        self.time_loop = 0
        self.playing = False
        self.looping = False
        self.ortho_camera = True
        self.ships = False
        self.view_curve = True
        self.phong = False
        
        #Velocidades de crecimiento de cada parametro
        self.speed = 0.3
        self.theta_speed = 3
        self.phi_speed = 2.5
        self.zoom_speed = 10
        self.fovi_speed = 40
        self.look_speed = 15
        self.step_speed = 150
        
        #Activadores
        self.front_switch = 0
        self.phi_switch = 0
        self.theta_switch = 0
        self.zoom_switch = 0
        self.fovi_switch = 0
        self.look_switch = 0

    def update(self, dt):
        if not self.playing and not self.looping:
            self.theta += self.theta_speed * self.theta_switch * dt
            self.phi += self.phi_speed * self.phi_switch * dt
            
            self.front[0] = mt.sin(self.theta) * mt.cos(self.phi) 
            self.front[1] = mt.sin(self.theta) * mt.sin(self.phi)
            self.front[2] = mt.cos(self.theta) 

            self.up[0] = - mt.cos(self.theta) * mt.cos(self.phi)
            self.up[1] = - mt.cos(self.theta) * mt.sin(self.phi)
            self.up[2] = mt.sin(self.theta)

            self.move = self.speed * self.front * self.front_switch

        elif self.playing:
            self.step += self.step_speed * dt

        self.zoom += self.zoom_speed * self.zoom_switch * dt
        self.time += dt
        
        self.fovi += self.fovi_speed * self.fovi_switch * dt
        self.look += self.look_speed * self.look_switch * dt
        
        
#Componentes de la escena
class scene_components:
    def __init__(self, pipeline, phong_pipeline):
        self.tex_params = [GL_REPEAT, GL_REPEAT, GL_NEAREST, GL_NEAREST]

        self.floor_shape = createGPUShape(pipeline, read_OBJ2(ASSETS["Floor"]))
        self.floor_shape.texture = sh.textureSimpleSetup(ASSETS["Floor_Texture"], *self.tex_params)

        self.pillar_shape = createGPUShape(pipeline, read_OBJ2(ASSETS["Pillar"]))
        self.pillar_shape.texture = sh.textureSimpleSetup(ASSETS["Pillar_Texture"], *self.tex_params)

        self.deer_shape = createGPUShape(pipeline, read_OBJ2(ASSETS["Deer"]))
        self.deer_shape.texture = sh.textureSimpleSetup(ASSETS["Deer_Texture"], *self.tex_params)

        self.deer_shape_shadow = createGPUShape(pipeline, read_OBJ2(ASSETS["Deer"]))
        self.deer_shape_shadow.texture = sh.textureSimpleSetup(ASSETS["Shadow_Texture"], *self.tex_params)

        self.portal_cube_shape = createGPUShape(pipeline, read_OBJ2(ASSETS["Portal_cube"]))
        self.portal_cube_shape.texture = sh.textureSimpleSetup(ASSETS["Portal_cube_Texture"], *self.tex_params)

        self.portal_cube_shape_shadow = createGPUShape(pipeline, read_OBJ2(ASSETS["Portal_cube"]))
        self.portal_cube_shape_shadow.texture = sh.textureSimpleSetup(ASSETS["Shadow_Texture"], *self.tex_params)
        
        self.star_shape = createGPUShape(pipeline, read_OBJ2(ASSETS["Star"]))
        self.star_shape.texture = sh.textureSimpleSetup(ASSETS["Star_Textures"], *self.tex_params)
    

        #PHONG Pipeline

        self.phong_floor_shape = createGPUShape(phong_pipeline, read_OBJ2(ASSETS["Floor"]))
        self.phong_floor_shape.texture = sh.textureSimpleSetup(ASSETS["Floor_Texture"], *self.tex_params)

        self.phong_pillar_shape = createGPUShape(phong_pipeline, read_OBJ2(ASSETS["Pillar"]))
        self.phong_pillar_shape.texture = sh.textureSimpleSetup(ASSETS["Pillar_Texture"], *self.tex_params)

        self.phong_deer_shape = createGPUShape(phong_pipeline, read_OBJ2(ASSETS["Deer"]))
        self.phong_deer_shape.texture = sh.textureSimpleSetup(ASSETS["Deer_Texture"], *self.tex_params)

        self.phong_deer_shape_shadow = createGPUShape(phong_pipeline, read_OBJ2(ASSETS["Deer"]))
        self.phong_deer_shape_shadow.texture = sh.textureSimpleSetup(ASSETS["Shadow_Texture"], *self.tex_params)

        self.phong_portal_cube_shape = createGPUShape(phong_pipeline, read_OBJ2(ASSETS["Portal_cube"]))
        self.phong_portal_cube_shape.texture = sh.textureSimpleSetup(ASSETS["Portal_cube_Texture"], *self.tex_params)

        self.phong_portal_cube_shape_shadow = createGPUShape(phong_pipeline, read_OBJ2(ASSETS["Portal_cube"]))
        self.phong_portal_cube_shape_shadow.texture = sh.textureSimpleSetup(ASSETS["Shadow_Texture"], *self.tex_params)

        self.phong_star_shape = createGPUShape(pipeline, read_OBJ2(ASSETS["Star"]))
        self.phong_star_shape.texture = sh.textureSimpleSetup(ASSETS["Star_Phong_Textures"], *self.tex_params)

#Clase que maniene toda la información de las curvas que generan las naves
class curve:
    def __init__(self, pipeline, points, joints, control_pipeline, control_p):

        self.NUM_POINTS = len(control_p)

        self.point_data = pipeline.vertex_list(
            len(points), GL_POINTS, position="f"
            )
        
        self.point_data.position[:] = tuple(
            chain(*(tuple(p) for p in points))
        )

        self.c_point_data = control_pipeline.vertex_list(
            len(control_p), GL_POINTS, position="f"
            )
        
        self.c_point_data.position[:] = tuple(
            chain(*(tuple(p) for p in control_p))
        )

        if len(points) >= 2:
            self.joint_data = pipeline.vertex_list_indexed(
                len(points), GL_LINES, tuple(chain(*(j for j in joints))), 
                position="f")

            self.joint_data.position[:] = tuple(
                chain(*(tuple(p) for p in points))
            )     

class Particle(object):
    def __init__(self, position, ttl, velocity):
        self.position = np.array(position, dtype=np.float32)
        self.ttl = ttl
        self.velocity = np.array(velocity, dtype=np.float32)

    def step(self, dt):
        self.ttl = self.ttl - dt
        self.position = self.position + dt*100 * self.velocity 

    def alive(self):
        return bool(self.ttl > 0)

class queue:
    def __init__(self):
        self.pair_list=[]
    def enq(self,x):
        self.pair_list.insert(0,x)
    def deq(self):
        assert len(self.pair_list)>0
        return self.pair_list.pop()
    def is_empty(self):
        return len(self.pair_list)==0


SCREEN = pyglet.canvas.Display().get_default_screen()
SCREEN_RATIO = SCREEN.width / SCREEN.height

WIDTH = 1100
HEIGHT = 780
WINDOW_RATIO = WIDTH / HEIGHT
FULL_SCREEN = False

window = Window(WIDTH, HEIGHT, "Tarea 4 Modelación", resizable=False)
window.set_fullscreen(FULL_SCREEN)

Shapes = scene_components(PIPELINE,PHONG_PIPELINE)
Ship1 = Ship(PIPELINE,PHONG_PIPELINE)

#Parametros de la escena
ALTURASHIPS = 1.5
ORTHO_SCALE = 15.0
FLOOR_SCALE = 10.0
PILLAR_POSITION = np.array([7.0, 7.0, 0.0], dtype = float)
DEER_SCALE = 4
DEER_POSITION = np.array([20.0, -20.0, 0.0])
#Cube
PORTAL_CUBE_POSITION = np.array([-22.0, 22.0, 10.0], dtype = float)
PORTAL_CUBE_SCALE = 0.7
CUBE_ANG_SPEED_h = 2
CUBE_ANG_SPEED_rotz = 3
CUBE_AMPLITUD = 2
#Star
STAR_ANGULAR_SPEED = 1.5
STAR_CENTER_ANGULAR_SPEED = 0.5
RADIOUS_STAR = 40
ALTURA_STAR = 10
SCALE_STAR = 3
#Cámara tercera persona
PERSPECTIVE_EYE = np.array([12.0, 0.0, 7.5, 0.0], dtype = float)
PERSPECTIVE_AT = np.array([3,0,0,0], dtype = float)
ANGLE = mt.atan(PERSPECTIVE_EYE[2]/PERSPECTIVE_EYE[0])
PERSPECTIVE_FOVI = 80
UP = np.array([0,0,ALTURASHIPS], dtype = float)
#Curva
CONTROL_POINTS = queue()
CONTROL_POINTS_TAN = queue()
N = 100
TAN_ESCALAR = 300
TAN_C_ESCALAR = 10
#Particles
PARTICLES = queue()
PARTICLES_DATA = None
POS_PARTICLES = np.array([-1,0,0,0], dtype = float)
#Naves secundarias
SHIPS2 = {"scale": 0.71, "back": 1.6, "vertical": 2.7}
SHIPS3 = {"scale": 0.57, "back": 3.0, "vertical": 5.0}

#Transformaciones de las naves secundarias
ship2_scale = tr.uniformScale(SHIPS2["scale"])
ship3_scale = tr.uniformScale(SHIPS3["scale"])
shadow2_scale = tr.scale(SHIPS2["scale"], SHIPS2["scale"], 1)
shadow3_scale = tr.scale(SHIPS3["scale"], SHIPS3["scale"], 1)

#Grafo de Escena

#GPU_Shapes
#Nave
GPU_ship = sg.SceneGraphNode("GPU_ship")
GPU_ship.childs += [Ship1.ex_shape]

GPU_shadow = sg.SceneGraphNode("GPU_shadow")
GPU_shadow.childs += [Ship1.shadow_shape]

#Cosas
GPU_floor = sg.SceneGraphNode("GPU_floor")
GPU_floor.transform = tr.scale(FLOOR_SCALE, FLOOR_SCALE, 1.0)
GPU_floor.childs += [Shapes.floor_shape]

GPU_pillar = sg.SceneGraphNode("GPU_pillar")
GPU_pillar.transform = tr.matmul([tr.uniformScale(2.5),  
                                  tr.rotationZ(mt.pi)])
GPU_pillar.childs += [Shapes.pillar_shape]

GPU_deer = sg.SceneGraphNode("GPU_deer")
GPU_deer.transform = tr.matmul([
    tr.translate(*DEER_POSITION), 
    tr.uniformScale(DEER_SCALE)
])
GPU_deer.childs += [Shapes.deer_shape]

GPU_deer_shadow = sg.SceneGraphNode("GPU_deer_shadow")
GPU_deer_shadow.transform = tr.matmul([
    tr.translate(*(DEER_POSITION + np.array([0, 0, 0.001]))), 
    tr.scale(1.0, 1.0, 0.01),
    tr.uniformScale(DEER_SCALE)
])
GPU_deer_shadow.childs = [Shapes.deer_shape_shadow]

GPU_portal_cube = sg.SceneGraphNode("GPU_portal_cube")
GPU_portal_cube.childs += [Shapes.portal_cube_shape]

GPU_portal_cube_shadow = sg.SceneGraphNode("GPU_portal_cube_shadow")
GPU_portal_cube_shadow.childs += [Shapes.portal_cube_shape_shadow]

GPU_star = sg.SceneGraphNode("GPU_star")
GPU_star.childs = [Shapes.star_shape]

#Phong Shapes
#Nave
GPU_phong_ship = sg.SceneGraphNode("GPU_phong_ship")
GPU_phong_ship.childs += [Ship1.phong_shape]

GPU_phong_shadow = sg.SceneGraphNode("GPU_phong_shadow")
GPU_phong_shadow.childs += [Ship1.phong_shadow_shape]

#Cosas
GPU_phong_floor = sg.SceneGraphNode("GPU_phong_floor")
GPU_phong_floor.transform = tr.scale(FLOOR_SCALE, FLOOR_SCALE, 1.0)
GPU_phong_floor.childs += [Shapes.phong_floor_shape]

GPU_phong_pillar = sg.SceneGraphNode("GPU_phong_pillar")
GPU_phong_pillar.transform = tr.matmul([tr.uniformScale(2.5),  
                                  tr.rotationZ(mt.pi)])
GPU_phong_pillar.childs += [Shapes.phong_pillar_shape]

GPU_phong_deer = sg.SceneGraphNode("GPU_phong_deer")
GPU_phong_deer.transform = tr.matmul([
    tr.translate(*DEER_POSITION), 
    tr.uniformScale(DEER_SCALE)
])
GPU_phong_deer.childs += [Shapes.phong_deer_shape]

GPU_phong_deer_shadow = sg.SceneGraphNode("GPU_phong_deer_shadow")
GPU_phong_deer_shadow.transform = tr.matmul([
    tr.translate(*(DEER_POSITION + np.array([0, 0, 0.001]))), 
    tr.scale(1.0, 1.0, 0.01),
    tr.uniformScale(DEER_SCALE)
])
GPU_phong_deer_shadow.childs = [Shapes.phong_deer_shape_shadow]

GPU_phong_portal_cube = sg.SceneGraphNode("GPU_phong_portal_cube")
GPU_phong_portal_cube.childs += [Shapes.phong_portal_cube_shape]

GPU_phong_portal_cube_shadow = sg.SceneGraphNode("GPU_phong_portal_cube_shadow")
GPU_phong_portal_cube_shadow.childs += [Shapes.phong_portal_cube_shape_shadow]

GPU_phong_star = sg.SceneGraphNode("GPU_phong_star")
GPU_phong_star.childs = [Shapes.phong_star_shape]

#Nodos con info

#Naves

naveLider = sg.SceneGraphNode("naveLider")
naveLider.childs += [GPU_ship]

nave_2D = sg.SceneGraphNode("nave_2D")
nave_2D.childs += [GPU_ship]

nave_2I = sg.SceneGraphNode("nave_2I")
nave_2I.childs += [GPU_ship]

nave_3D = sg.SceneGraphNode("nave_3D")
nave_3D.childs += [GPU_ship]

nave_3I = sg.SceneGraphNode("nave_3I")
nave_3I.childs += [GPU_ship]

naves = sg.SceneGraphNode("naves")
naves.childs += [naveLider]

#Sombras de las naves
shadow_lider = sg.SceneGraphNode("shadow_lider")
shadow_lider.childs += [GPU_shadow]

shadow_2D = sg.SceneGraphNode("shadow_2D")
shadow_2D.childs += [GPU_shadow]

shadow_2I = sg.SceneGraphNode("shadow_2I")
shadow_2I.childs += [GPU_shadow]

shadow_3D = sg.SceneGraphNode("shadow_3D")
shadow_3D.childs += [GPU_shadow]

shadow_3I = sg.SceneGraphNode("shadow_3I")
shadow_3I.childs += [GPU_shadow]

shadows = sg.SceneGraphNode("shadows")
shadows.childs += [shadow_lider]

#Pilares
pillar1 = sg.SceneGraphNode("pillar1")
pillar1.transform = tr.translate(PILLAR_POSITION[0], PILLAR_POSITION[1] + 1, 0)
pillar1.childs += [GPU_pillar]

pillar2 = sg.SceneGraphNode("pillar2")
pillar2.transform = tr.translate(PILLAR_POSITION[0] + 1, -PILLAR_POSITION[1] + 1, 0)
pillar2.childs += [GPU_pillar]

pillar3 = sg.SceneGraphNode("pillar3")
pillar3.transform = tr.translate(-PILLAR_POSITION[0], PILLAR_POSITION[1] - 1, 0)
pillar3.childs += [GPU_pillar]

pillar4 = sg.SceneGraphNode("pillar4")
pillar4.transform = tr.translate(-PILLAR_POSITION[0] + 1, -PILLAR_POSITION[1] - 2, 0)
pillar4.childs += [GPU_pillar]

pillares =sg.SceneGraphNode("pillares")
pillares.childs += [pillar1]
pillares.childs += [pillar2]
pillares.childs += [pillar3]
pillares.childs += [pillar4]

#Portal Cube
cube_movement = sg.SceneGraphNode("cube_movement")
cube_movement.childs += [GPU_portal_cube]

cube_shadow_movement = sg.SceneGraphNode("cube_shadow_movement")
cube_shadow_movement.childs += [GPU_portal_cube_shadow]

cube_portal = sg.SceneGraphNode("cube_portal")
cube_portal.childs += [cube_movement]
cube_portal.childs += [cube_shadow_movement]

#Estrella
star = sg.SceneGraphNode("star")
star.childs = [GPU_star]

#Nodo principal
origen = sg.SceneGraphNode("origen")
origen.childs += [shadows]
origen.childs += [GPU_floor]
origen.childs += [naves]
origen.childs += [pillares]
origen.childs += [GPU_deer]
origen.childs += [GPU_deer_shadow]
origen.childs += [cube_portal]
origen.childs += [star]

#Canciones
Song_1 = pyglet.media.load(ASSETS["Song_1"])
Song_2 = pyglet.media.load(ASSETS["Song_2"])
Player = pyglet.media.Player()
SONGS = queue(); SONGS.enq(Song_1), SONGS.enq(Song_2)
Player.queue(Song_1)
Player.queue(Song_2)

#Preparando los Pipelines

glEnable(GL_DEPTH_TEST)

glUseProgram(PIPELINE.shaderProgram)

model = tr.rotationX(mt.pi/2)

glUniformMatrix4fv(glGetUniformLocation(PIPELINE.shaderProgram, "model"), 1, GL_TRUE, model)

glUseProgram(PHONG_PIPELINE.shaderProgram)

glUniformMatrix4fv(glGetUniformLocation(PHONG_PIPELINE.shaderProgram, "model"), 1, GL_TRUE, model)
glUniform3f(glGetUniformLocation(PHONG_PIPELINE.shaderProgram, "material.ambient"), 0.2, 0.2, 0.2)
glUniform3f(glGetUniformLocation(PHONG_PIPELINE.shaderProgram, "material.diffuse"), 0.9, 0.9, 0.9)
glUniform3f(glGetUniformLocation(PHONG_PIPELINE.shaderProgram, "material.specular"), 1.0, 1.0, 1.0)
glUniform1f(glGetUniformLocation(PHONG_PIPELINE.shaderProgram, "material.shininess"), 10)

#Determinar el tamaño de las particulas
pipeline_particles.use()
pipeline_particles["max_ttl"] = 3

#Elementos para crear la curva
C_POINTS,POINTS,JOINTS,TANGENTES,THETA,PHI = [],[],[],[],[],[]
CURVE,curve_was_made,contador= None,False,0

#Configuraciones distintas de la fuente de luz, modificando las 6 contantes del modelo de Phong y la velocidad de esta misma
CONFIG_i = 0
contador2 = CONFIG_i
AMBIENT,DIFFUSE,SPECULAR,CONSTANT,LINEAR,QUADRATIC = 0.35,1.35,-1.3877787807814457e-17,-0.05,0.025,0.0

CONFIG_0 = [0.35,1.35,-1.3877787807814457e-17,-0.05,0.025,0.0,1.5]
CONFIG_1 = [0.15,1.35,-1.3877787807814457e-17,-1.3877787807814457e-17,0.05,1.734723475976807e-18,5.5]
CONFIG_2 = [0.05,1.35,-1.3877787807814457e-17,-1.3877787807814457e-17,0.05,1.734723475976807e-18,11.40]
CONFIG_3 = [-0.2,1.35,-1.3877787807814457e-17,-0.65,-0.025,0.01,31.95]
CONFIG_4 = [-0.2,1.35,-1.3877787807814457e-17,1.1,-0.3375,0.035,105.90000000000128]
CONFIG_5 = [-4.2,1.5,2.914335439641036e-16,0.05,1.734723475976807e-17,-1.734723475976807e-18,0]
CONFIG = np.array([CONFIG_0, CONFIG_1, CONFIG_2, CONFIG_3, CONFIG_4, CONFIG_5])

#Seguimiento de Inputs 
@window.event
def on_key_press(symbol, modifiers):
    global C_POINTS,JOINTS,POINTS,TANGENTES,THETA,PHI
    global CURVE,curve_was_made,FULL_SCREEN,contador
    global AMBIENT,DIFFUSE,SPECULAR,CONSTANT,LINEAR,QUADRATIC,STAR_ANGULAR_SPEED,contador2
    global SONGS

    #movimiento de la nave
    if symbol == pw.key.A:
        Ship1.phi_switch += 1.0
    if symbol == pw.key.D:
        Ship1.phi_switch -= 1.0
    if symbol == pw.key.W:
        Ship1.front_switch += 1.0
    if symbol == pw.key.S:
        Ship1.front_switch -= 1.0  

    #Implementaciones
    if symbol == pw.key.C:
        Ship1.ortho_camera = not Ship1.ortho_camera
    if symbol == pw.key.Q:
        Ship1.ships = not Ship1.ships
    if symbol == pw.key.F:
        FULL_SCREEN = not FULL_SCREEN
        window.set_fullscreen(FULL_SCREEN)
    if symbol == pw.key._6:
        STAR_ANGULAR_SPEED -= 0.15
    if symbol == pw.key._7:
        STAR_ANGULAR_SPEED += 0.15

    #Grabación de curva
    if symbol == pw.key.R:
        CONTROL_POINTS.enq(np.array([np.copy(Ship1.pos + UP)]).T)
        CONTROL_POINTS_TAN.enq(np.array([np.copy(TAN_C_ESCALAR * Ship1.front)]).T)
        POINT_N = len(CONTROL_POINTS.pair_list) 
        if POINT_N == 1:
            C_POINTS = CONTROL_POINTS.pair_list[0].T
            POINTS = CONTROL_POINTS.pair_list[0].T
        elif POINT_N == 2:
            P1 = CONTROL_POINTS.deq()
            P2 = CONTROL_POINTS.pair_list[0]
            T1 = CONTROL_POINTS_TAN.deq()
            T2 = CONTROL_POINTS_TAN.pair_list[0]
            CURVA = cv.hermiteMatrix(P1,P2,T1,T2)
            HERMITE = cv.evalCurve(CURVA,N)
            if len(POINTS) == 1:
                POINTS = HERMITE
                C_POINTS = [HERMITE[0], HERMITE[N-1]]
            else:
                POINTS = np.concatenate((POINTS,HERMITE), axis=0)
                C_POINTS.append(HERMITE[N-1])
               
            for i in range(N-1):
                JOINTS.append([i+N*(contador),i+1+N*(contador)])
            contador += 1
            for i in range(N):
                if i < N-2:
                    tan_i = np.array(HERMITE[i+2]) - np.array(HERMITE[i])
                else:
                    tan_i = np.array(HERMITE[N-1]) - np.array(HERMITE[N-2])
                tan_i = np.multiply(tan_i,TAN_ESCALAR)
                norma_i = mt.sqrt(tan_i[0]**2 + tan_i[1]**2 + tan_i[2]**2)
                tan_i = list(np.multiply(tan_i,1/norma_i))
                TANGENTES.append(tan_i)
                THETA.append(mt.atan2(mt.sqrt(tan_i[0]**2 + tan_i[1]**2),tan_i[2]))
                PHI.append(mt.atan2(tan_i[1],tan_i[0]))
                    
        CURVE = curve(pipeline_curve, POINTS, JOINTS, pipeline_control_points, C_POINTS)
        if not curve_was_made:
            curve_was_made = True 

    #Reproduccionde la curva
    if symbol == pw.key._1:
        if curve_was_made and len(C_POINTS) > 1:
            if Ship1.playing: 
                Ship1.step = 0
            Ship1.playing = not Ship1.playing
    
    #Borrar curva existente
    if symbol == pw.key.TAB:
        if Ship1.playing: 
            Ship1.playing = not Ship1.playing
        C_POINTS,POINTS,JOINTS,TANGENTES,THETA,PHI = [],[],[],[],[],[]
        CURVE,curve_was_made,contador,Ship1.step = None,False,0,0
        CONTROL_POINTS.deq()
        CONTROL_POINTS_TAN.deq()
    
    #Calibrar fuente de luz
    if symbol == pw.key.N: 
        i = contador2 % len(CONFIG)
        [AMBIENT,DIFFUSE,SPECULAR,CONSTANT,LINEAR,QUADRATIC,STAR_ANGULAR_SPEED] = CONFIG[i]
        contador2 += 1
    
    #Calibrar Nave
    if symbol == pw.key.E:
        if [Ship1.theta,Ship1.phi] != [mt.pi/2,0]:
            Ship1.theta,Ship1.phi = mt.pi/2,0

    #Cambio de Pipeline      
    if symbol == pw.key.SPACE:
        Ship1.phong = not Ship1.phong
        if contador2 == CONFIG_i:
            [AMBIENT,DIFFUSE,SPECULAR,CONSTANT,LINEAR,QUADRATIC,STAR_ANGULAR_SPEED] = CONFIG[CONFIG_i]
        contador2 += 1     
        Player.next_source()
        next_song = SONGS.deq()
        Player.queue(next_song)
        SONGS.enq(next_song)


    #Visuzalizacion de la curva
    if symbol == pw.key.V:
        Ship1.view_curve = not Ship1.view_curve

    #zoom
    if symbol == pw.key.X:
        if Ship1.ortho_camera:
            Ship1.zoom_switch -= 1.0
        else:
            Ship1.fovi_switch += 1.0
            Ship1.look_switch -= 1.0
    if symbol == pw.key.Z:
        if Ship1.ortho_camera:
            Ship1.zoom_switch += 1.0
        else:
            Ship1.fovi_switch -= 1.0
            Ship1.look_switch += 1.0

    #Calibrar parametros de la fuente de luz
    if symbol == pw.key.K:
        AMBIENT -= 0.05
    if symbol == pw.key.L:
        DIFFUSE -= 0.05
    if symbol == pw.key.MINUS:
        SPECULAR -= 0.05
    if symbol == pw.key.G:
        CONSTANT -= 0.05
    if symbol == pw.key.H:
        LINEAR -= 0.0125
    if symbol == pw.key.J:
        QUADRATIC -= 0.005

    if symbol == pw.key.I:
        AMBIENT += 0.05
    if symbol == pw.key.O:
        DIFFUSE += 0.05
    if symbol == pw.key.P:
        SPECULAR += 0.05
    if symbol == pw.key.T:
        CONSTANT += 0.05
    if symbol == pw.key.Y:
        LINEAR += 0.0125
    if symbol == pw.key.U:
        QUADRATIC += 0.005

    if symbol == pw.key.M:
        print("AMBIENT:", AMBIENT)
        print("DIFFUSE:", DIFFUSE)
        print("SPECULAR:", SPECULAR)
        print("CONSTANT:", CONSTANT)
        print("LINEAR:", LINEAR)
        print("QUADRATIC:", QUADRATIC)
        print("ANGULAR_SPEED:", STAR_ANGULAR_SPEED)


@window.event
def on_key_release(symbol, modifiers):
    if symbol == pw.key.A:
        Ship1.phi_switch -= 1.0
    if symbol == pw.key.D:
        Ship1.phi_switch += 1.0
    if symbol == pw.key.W:
        Ship1.front_switch -= 1.0
    if symbol == pw.key.S:
        Ship1.front_switch += 1.0

    #zoom
    if symbol == pw.key.X:
        if Ship1.ortho_camera:
            Ship1.zoom_switch += 1.0
        else:
            Ship1.fovi_switch -= 1.0
            Ship1.look_switch += 1.0
    if symbol == pw.key.Z:
        if Ship1.ortho_camera:
            Ship1.zoom_switch -= 1.0 
        else:
            Ship1.fovi_switch += 1.0
            Ship1.look_switch -= 1.0  

@window.event 
def on_mouse_motion(x,y,dx,dy):
    if dy > 0:
        Ship1.theta_switch = 1.0
    elif dy < 0:
        Ship1.theta_switch = -1.0   


def update(dt):
    Ship1.update(dt)

#Dibujado de la escena
@window.event
def on_draw():
    window.clear()
    
    glEnable(GL_DEPTH_TEST)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glEnable(GL_PROGRAM_POINT_SIZE)

    glClearColor(0.0, 0.0, 0.0, 1.0)
    
    #Movimiento del cubo
    cube_movement.transform = tr.matmul([tr.translate(*PORTAL_CUBE_POSITION), 
                                        tr.translate(0, 0, CUBE_AMPLITUD * mt.sin(CUBE_ANG_SPEED_h * Ship1.time)),
                                        tr.rotationZ(CUBE_ANG_SPEED_rotz * Ship1.time), 
                                        tr.uniformScale(PORTAL_CUBE_SCALE)])

    cube_shadow_movement.transform = tr.matmul([tr.translate(PORTAL_CUBE_POSITION[0], PORTAL_CUBE_POSITION[1], 0.1), 
                                        tr.rotationZ(CUBE_ANG_SPEED_rotz * Ship1.time), 
                                        tr.scale(PORTAL_CUBE_SCALE, PORTAL_CUBE_SCALE, 0.000001)])
    
    #Movimiento de la estrella
    star.transform = tr.matmul([tr.translate(RADIOUS_STAR * mt.cos(STAR_ANGULAR_SPEED * Ship1.time), RADIOUS_STAR * mt.sin(STAR_ANGULAR_SPEED * Ship1.time), ALTURA_STAR),
                                        tr.rotationZ(STAR_CENTER_ANGULAR_SPEED * Ship1.time), 
                                        tr.uniformScale(SCALE_STAR)])

    #Dibujar naves secundarias
    if Ship1.ships:
        naves.childs = [naveLider, nave_2D, nave_2I, nave_3D, nave_3I]
        shadows.childs = [shadow_lider,shadow_2D, shadow_2I, shadow_3D, shadow_3I]
    else:   
        naves.childs = [naveLider]
        shadows.childs = [shadow_lider]

    #Transformaciones para las  naves
    if Ship1.playing:
        #especificaciones de Step
        STEP = int(Ship1.step)
        if STEP >= len(POINTS)-1:
            Ship1.step = 0
            STEP = int(Ship1.step)

        ship_Transform = tr.matmul([
            tr.translate(*POINTS[STEP]), 
            tr.rotationZ(PHI[STEP]),
            tr.rotationY(THETA[STEP] - mt.pi/2)
        ])

        Ship1.pos = np.array([POINTS[STEP,0], POINTS[STEP,1], POINTS[STEP,2] - ALTURASHIPS])
        Ship1.phi = PHI[STEP]
        Ship1.theta = THETA[STEP]
    else:
        ship_Transform = tr.matmul([
            tr.translate(0, 0, ALTURASHIPS),
            tr.translate(*Ship1.move),
            tr.translate(*Ship1.pos),
            tr.rotationZ(Ship1.phi),
            tr.rotationY(Ship1.theta - mt.pi/2)
        ])
    
    naveLider.transform = ship_Transform

    nave_2D.transform = tr.matmul([
            tr.translate(-SHIPS2["back"], -SHIPS2["vertical"], 0),
            ship_Transform,
            ship2_scale
        ])
    
    nave_2I.transform = tr.matmul([
            tr.translate(-SHIPS2["back"], SHIPS2["vertical"], 0),
            ship_Transform,
            ship2_scale
        ])
    
    nave_3D.transform = tr.matmul([
            tr.translate(-SHIPS3["back"], -SHIPS3["vertical"], 0),
            ship_Transform,
            ship3_scale
        ])
    
    nave_3I.transform = tr.matmul([
            tr.translate(-SHIPS3["back"], SHIPS3["vertical"], 0),
            ship_Transform,
            ship3_scale
        ])
    
    #Transformaciones para las sombras de las naves
    shadow_Transform = tr.matmul([
            tr.translate(Ship1.pos[0] + Ship1.move[0], Ship1.pos[1] + Ship1.move[1], 0.1),
            tr.scale(1.0, 1.0, 0.01),
            tr.rotationZ(Ship1.phi),
            tr.rotationY(Ship1.theta - mt.pi/2),
        ])
    
    shadow_lider.transform = shadow_Transform

    shadow_2D.transform = tr.matmul([
            tr.translate(-SHIPS2["back"], -SHIPS2["vertical"], 0), 
            shadow_Transform,
            shadow2_scale
        ])
    
    shadow_2I.transform = tr.matmul([
            tr.translate(-SHIPS2["back"], SHIPS2["vertical"], 0), 
            shadow_Transform,
            shadow2_scale
        ])
    
    shadow_3D.transform = tr.matmul([
            tr.translate(-SHIPS3["back"], -SHIPS3["vertical"], 0), 
            shadow_Transform,
            shadow3_scale
        ])
    
    shadow_3I.transform = tr.matmul([
            tr.translate(-SHIPS3["back"], SHIPS3["vertical"], 0), 
            shadow_Transform,
            shadow3_scale
        ])
    
    #Actualizar la posición de las nave
    Ship1.pos += Ship1.move

    #Cámaras
    if Ship1.ortho_camera:
        view = tr.lookAt(
            np.array([-100.0, -100.0, 50.0], dtype = float) + Ship1.pos + np.array([0.0, 0.0, ALTURASHIPS], dtype = float), 
            Ship1.pos + UP, 
            np.array([0.0, 0.0, 1.0])
        )
        RECTANGLE = np.array([-1.41, 1.41, -1, 1])
        if FULL_SCREEN:
            RECTANGLE[0],RECTANGLE[1] = -SCREEN_RATIO,SCREEN_RATIO
        ortho_param1 = (ORTHO_SCALE + Ship1.zoom) * RECTANGLE
        projection1 = tr.ortho(*ortho_param1, 0.1, 500)
    else:
        Rotation = tr.matmul([tr.rotationZ(Ship1.phi),
                        tr.rotationY(Ship1.theta - mt.pi/2)])

        dynamic_eye = np.array([- PERSPECTIVE_EYE[0] - Ship1.look * mt.cos(ANGLE), 
                                0, PERSPECTIVE_EYE[2] + Ship1.look * mt.sin(ANGLE), 0],dtype = float)
        
        #Límites de la cámara en tercera persona
        if Ship1.fovi < 0:
            Ship1.fovi = 0
            Ship1.look = 0
            dynamic_eye[0] = -PERSPECTIVE_EYE[0]
            dynamic_eye[2] = PERSPECTIVE_EYE[2]
        elif Ship1.fovi > 39.957:
            Ship1.fovi = 39.957
            Ship1.look = -14.98    
            dynamic_eye[0] = -PERSPECTIVE_EYE[0] - Ship1.look * mt.cos(ANGLE)
            dynamic_eye[2] = PERSPECTIVE_EYE[2] + Ship1.look * mt.sin(ANGLE)
           
        RATIO = WINDOW_RATIO
        if FULL_SCREEN:
            RATIO = SCREEN_RATIO

        eye = Ship1.pos + UP + tr.matmul([Rotation, dynamic_eye])[0:3]
        
        at = Ship1.pos + UP + tr.matmul([Rotation, PERSPECTIVE_AT])[0:3]
        
        up = tr.matmul([ship_Transform, np.array([0,0,1,0], dtype = float)])[0:3]

        projection1 = tr.perspective(PERSPECTIVE_FOVI + Ship1.fovi, RATIO, 0.001, 500)

        view = tr.lookAt(eye,at,up)
    
    glUseProgram(PIPELINE.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(PIPELINE.shaderProgram, "projection"), 1, GL_TRUE, projection1)
    glUniformMatrix4fv(glGetUniformLocation(PIPELINE.shaderProgram, "view"), 1, GL_TRUE, view)

    glUseProgram(PHONG_PIPELINE.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(PHONG_PIPELINE.shaderProgram, "projection"), 1, GL_TRUE, projection1)
    glUniformMatrix4fv(glGetUniformLocation(PHONG_PIPELINE.shaderProgram, "view"), 1, GL_TRUE, view)

    #Resetear ángulo del mouse
    Ship1.theta_switch = 0

    #Cambio de Pipeline
    if Ship1.phong:
        glUseProgram(PHONG_PIPELINE.shaderProgram)
        naveLider.childs = [GPU_phong_ship]
        nave_2D.childs = [GPU_phong_ship]
        nave_2I.childs = [GPU_phong_ship]
        nave_3D.childs = [GPU_phong_ship]
        nave_3I.childs = [GPU_phong_ship]

        shadow_lider.childs = [GPU_phong_shadow]
        shadow_2D.childs = [GPU_phong_shadow]        
        shadow_2I.childs = [GPU_phong_shadow]
        shadow_3D.childs = [GPU_phong_shadow]
        shadow_3I.childs = [GPU_phong_shadow]

        pillar1.childs = [GPU_phong_pillar]
        pillar2.childs = [GPU_phong_pillar]        
        pillar3.childs = [GPU_phong_pillar]
        pillar4.childs = [GPU_phong_pillar]

        cube_movement.childs = [GPU_phong_portal_cube]
        cube_shadow_movement.childs = [GPU_phong_portal_cube_shadow]

        origen.childs = [shadows, GPU_phong_floor, naves, pillares, GPU_phong_deer, GPU_phong_deer_shadow, cube_portal]

        POINT_LIGHT_1 = [sh.PointLight([RADIOUS_STAR * mt.cos(STAR_ANGULAR_SPEED * Ship1.time),
                                         RADIOUS_STAR * mt.sin(STAR_ANGULAR_SPEED * Ship1.time),
                                          ALTURA_STAR], [1.0]*3, [0.9]*3, [1.0]*3,CONSTANT,LINEAR,QUADRATIC),
                                          sh.PointLight([30.0, 30.0, 5.0], [1.0]*3, [0.9]*3, [1.0]*3,CONSTANT,LINEAR,QUADRATIC)]
        sh.update_lights(POINT_LIGHT_1,PHONG_PIPELINE,AMBIENT,DIFFUSE,SPECULAR)

        sg.drawSceneGraphNode(origen, PHONG_PIPELINE, "transform")

        glUseProgram(PIPELINE.shaderProgram)

        star.childs = [GPU_phong_star]

        sg.drawSceneGraphNode(star, PIPELINE, "transform")
    else:
        glUseProgram(PIPELINE.shaderProgram)
        naveLider.childs = [GPU_ship]
        nave_2D.childs = [GPU_ship]
        nave_2I.childs = [GPU_ship]
        nave_3D.childs = [GPU_ship]
        nave_3I.childs = [GPU_ship]

        shadow_lider.childs = [GPU_shadow]
        shadow_2D.childs = [GPU_shadow]        
        shadow_2I.childs = [GPU_shadow]
        shadow_3D.childs = [GPU_shadow]
        shadow_3I.childs = [GPU_shadow]

        pillar1.childs = [GPU_pillar]
        pillar2.childs = [GPU_pillar]        
        pillar3.childs = [GPU_pillar]
        pillar4.childs = [GPU_pillar]

        cube_movement.childs = [GPU_portal_cube]
        cube_shadow_movement.childs = [GPU_portal_cube_shadow]

        star.childs = [GPU_star]

        origen.childs = [shadows, GPU_floor, naves, pillares, GPU_deer, GPU_deer_shadow, cube_portal, star]

        sg.drawSceneGraphNode(origen, PIPELINE, "transform")

    #Dibujar Trayectoria grabada
    if Ship1.view_curve and curve_was_made:
        pipeline_curve.use()
        pipeline_curve["projection"] = projection1.reshape(16, 1, order="F")
        pipeline_curve["view"] = view.reshape(16, 1, order="F")

        if CURVE.NUM_POINTS >= 2:
            CURVE.point_data.draw(GL_POINTS)
            CURVE.joint_data.draw(GL_LINES)

        pipeline_control_points.use()
        pipeline_control_points["projection"] = projection1.reshape(16, 1, order="F")
        pipeline_control_points["view"] = view.reshape(16, 1, order="F")

        if CURVE.NUM_POINTS >=1:
            CURVE.c_point_data.draw(GL_POINTS)

Player.play()
clock.schedule(update)
run()