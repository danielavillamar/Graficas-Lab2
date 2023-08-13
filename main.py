from gl import Renderer, Model
import shaders
import random

width = 900
height = 500
rend = Renderer(width, height)
# Color del Background
rend.glClearColor(255, 128, 192)
rend.glClear()

# Shaders fijos
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

# Renderizar objeto 2 con shader 2
objeto_shader2 = Model(
    "models/model.obj",
    translate=(0, -1, -10),
    rotate=(0, 180, 0),
    scale=(0.16, 0.16, 0.16),
)
# Se aplican condiciones de textura y shader a renderizar
objeto_shader2.LoadTexture("textures/model.bmp")
objeto_shader2.setShaders(shaders.vertexShader, shaders.matrixShader) 

# Renderizar objeto 3 con shader 3
objeto_shader3 = Model(
    "models/model.obj",
    translate=(5, -1, -10),
    rotate=(0, 180, 0),
    scale=(0.16, 0.16, 0.16),
)
# Se aplican condiciones de textura y shader a renderizar
objeto_shader3.LoadTexture("textures/model.bmp")
objeto_shader3.setShaders(shaders.vertexShader, shaders.rayTracingShader)

# Render final de los 3 objetos en pantalla
rend.glAddModel(objeto_shader1)
rend.glAddModel(objeto_shader2)
rend.glAddModel(objeto_shader3)

rend.glRender()

rend.glFinish("output.bmp")
