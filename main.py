from gl import Renderer, Model
import shaders
import random

width = 900
height = 500
rend = Renderer(width, height)

rend.glClearColor(255, 128, 192)
rend.glClear()


rend.vertexShader = shaders.vertexShader
rend.fragmentShader = shaders.fragmentShader

# Renderizar objeto 1 con shader 1
objeto_shader1 = Model(
    "models/model.obj",
    translate=(-5, -1, -10),
    rotate=(0, 180, 0),
    scale=(0.16, 0.16, 0.16),
)
# Se aplican condiciones de textura y shader a renderizar
objeto_shader1.LoadTexture("textures/model.bmp")
objeto_shader1.setShaders(shaders.vertexShader, shaders.burnedShader)

rend.glAddModel(objeto_shader1)

rend.glRender()

rend.glFinish("output.bmp")
