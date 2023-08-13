import numpy as np


def vertexShader(vertex, **kwargs):
    modelMatrix = kwargs["modelMatrix"]
    viewMatrix = kwargs["viewMatrix"]
    projectionMatrix = kwargs["projectionMatrix"]
    vpMatrix = kwargs["vpMatrix"]

    vt = [vertex[0], vertex[1], vertex[2], 1]

    vt = vpMatrix * projectionMatrix * viewMatrix * modelMatrix @ vt

    vt = vt.tolist()[0]

    vt = [vt[0] / vt[3], vt[1] / vt[3], vt[2] / vt[3]]

    return vt

def fragmentShader(**kwargs):
    tA, tB, tC = kwargs["texCoords"]
    texture = kwargs["texture"]
    u, v, w = kwargs["bCoords"]

    b = 1.0
    g = 1.0
    r = 1.0

    if texture != None:
        tU = u * tA[0] + v * tB[0] + w * tC[0]
        tV = u * tA[1] + v * tB[1] + w * tC[1]

        textureColor = texture.getColor(tU, tV)
        b *= textureColor[2]
        g *= textureColor[1]
        r *= textureColor[0]
    return r, g, b

def burnedShader(**kwargs):
    """
    Crea un efecto de quemado en el objeto.

    Args:
        texture: La textura del objeto.
        texCoords: Las coordenadas de textura para el fragmento actual.
        normals: Las normales del objeto.
        dLight: La dirección de la luz.
        bCoords: Las coordenadas de la cámara para el fragmento actual.

    Returns:
        Un color para el fragmento actual.
    """

    texture = kwargs["texture"]
    tA, tB, tC = kwargs["texCoords"]
    nA, nB, nC = kwargs["normals"]
    dLight = kwargs["dLight"]
    u, v, w = kwargs["bCoords"]

    # Obtiene el color base del objeto.
    b = 1.0
    g = 1.0
    r = 1.0

    if texture != None:
        tU = u * tA[0] + v * tB[0] + w * tC[0]
        tV = u * tA[1] + v * tB[1] + w * tC[1]

        textureColor = texture.getColor(tU, tV)
        b *= textureColor[2]
        g *= textureColor[1]
        r *= textureColor[0]

    # Calcula la intensidad de la luz.
    normal = [
        u * nA[0] + v * nB[0] + w * nC[0],
        u * nA[1] + v * nB[1] + w * nC[1],
        u * nA[2] + v * nB[2] + w * nC[2],
    ]

    dLight = np.array(dLight)
    intensity = np.dot(normal, -dLight)

    # Aplica el efecto de quemado.
    b *= intensity
    g *= intensity
    r *= intensity

    red = (1, 0, 0)

    b *= red[2]
    g *= red[1]
    r *= red[0]

    # Devuelve el color final del fragmento.
    if intensity > 0:
        return r, g, b
    else:
        return [0, 0, 0]

def matrixShader(texCoords, bCoords):
    """
    Crea un tablero de pequeños cuadros rosado y amarillo. Como una matriz o incluso un tablero de ajedrez.

    Args:
        texCoords: Las coordenadas de textura para el fragmento actual.
        bCoords: Las coordenadas de la cámara para el fragmento actual.

    Returns:
        Un color para el fragmento actual.
    """

    size = 45  # Ajuste el tamaño de los cuadrados, se puede ajustar a preferencia :)

    tU = texCoords[0] * bCoords[0] + texCoords[1] * bCoords[1]
    tV = texCoords[0] * bCoords[2] + texCoords[1] * bCoords[3]

    u = int(tU * size) % 2
    v = int(tV * size) % 2

    color = (1, 0.7, 0.7) if u ^ v else (0.9, 0.9, 0.7)

    return color


def caricatureShader(**kwargs):
    """
    Crea un efecto de caricatura en el objeto.

    Args:
        texture: La textura del objeto.
        texCoords: Las coordenadas de textura para el fragmento actual.
        normals: Las normales del objeto.
        dLight: La dirección de la luz.
        bCoords: Las coordenadas de la cámara para el fragmento actual.

    Returns:
        Un color para el fragmento actual.
    """

    # Obtiene el color base del objeto.
    b = 1.0
    g = 1.0
    r = 1.0

    if texture is not None:
        tU = u * tA[0] + v * tB[0] + w * tC[0]
        tV = u * tA[1] + v * tB[1] + w * tC[1]

        textureColor = texture.getColor(tU, tV)
        b *= textureColor[2]
        g *= textureColor[1]
        r *= textureColor[0]

    # No hay sombras en el sombreado de caricatura.
    intensity = 0.2

    # Toon shading bands
    # Los colores se mapean a diferentes intensidades para crear un efecto de "bandas" de toon.
    if intensity < 0.25:
        intensity = 0.2
    elif intensity < 0.5:
        intensity = 0.45
    elif intensity < 0.75:
        intensity = 0.7
    else:
        intensity = 0.95

    # No es necesario multiplicar por la intensidad, ya que se aplica el sombreado de toon.

    b = textureColor[2]
    g = textureColor[1]
    r = textureColor[0]

    # Agrega un contorno
    # El contorno se crea restando un valor constante del color base.
    outline = 2.0
    b = min(b - outline, 1.0)
    g = min(g - outline, 1.0)
    r = min(r - outline, 1.0)

    # Agrega algo de ruido
    # El ruido se agrega al color base para crear un efecto más aleatorio.
    noise = np.random.rand() * 0.1
    b += noise
    g += noise
    r += noise

    return r, g, b


def rayTracingShader(**kwargs):
    """
    Calcueta el color de un fragmento usando un algoritmo de "trazado de rayos" o mejor dicho rayos X.

    Args:
        texture: La textura del objeto.
        texCoords: Las coordenadas de textura para el fragmento actual.
        normals: Las normales del objeto.
        dLight: La dirección de la luz.
        bCoords: Las coordenadas de la cámara para el fragmento actual.

    Returns:
        Un color para el fragmento actual.
    """

    texture = kwargs["texture"]
    tA, tB, tC = kwargs["texCoords"]
    nA, nB, nC = kwargs["normals"]
    dLight = kwargs["dLight"]
    u, v, w = kwargs["bCoords"]
    
    # Calcula la dirección del rayo a través del objeto.
    normal=[u * nA[0] + v * nB[0] + w * nC[0],
            u * nA[1] + v * nB[1] + w * nC[1],
            u * nA[2] + v * nB[2] + w * nC[2]]
    
    # Calcula el color del fragmento.
    color = [1, 1, 1]
    
    if texture != None:
        tU = tA[0] * u + tB[0] * v + tC[0] * w
        tV = tA[1] * u + tB[1] * v + tC[1] * w
        textureColor = texture.getColor(tU, tV)
        color = [1 - c * t for c, t in zip(color, textureColor)]
    # Devuelve el color del fragmento.
    return color






