import struct
from collections import namedtuple
import mathLib as ml
import numpy as np
from obj import Obj
from math import pi, sin, cos, tan
from texture import Texture
from mathLib import barycentricCoords

V2 = namedtuple("Point2", ["x", "y"])
V3 = namedtuple("Point2", ["x", "y", "z"])

POINTS = 0
LINES = 1
TRIANGLES = 2
QUADS = 3


def char(c):
    # 1 byte
    return struct.pack("=c", c.encode("ascii"))


def word(w):
    # 2 bytes
    return struct.pack("=h", w)


def dword(d):
    # 4 bytes
    return struct.pack("=l", d)


def color(r, g, b):
    b = int(b * 255)
    g = int(g * 255)
    r = int(r * 255)

    if r > 255:
        r = 255
    elif r < 0:
        r = 0

    if g > 255:
        g = 255
    elif g < 0:
        g = 0

    if b > 255:
        b = 255
    elif b < 0:
        b = 0

    return bytes([b, g, r])


class Model(object):
    def __init__(
        self, fileName, translate=(0, 0, 0), rotate=(0, 0, 0), scale=(1, 1, 1)
    ):
        model = Obj(fileName)

        self.vertices = model.vertices
        self.texcoords = model.texcoords
        self.normals = model.normals
        self.faces = model.faces

        self.translate = translate
        self.rotate = rotate
        self.scale = scale

        self.setShaders(None, None)

    def LoadTexture(self, filename):
        self.texture = Texture(filename)

    def setShaders(self, vertexShader, fragmentShader):
        self.vertexShader = vertexShader
        self.fragmentShader = fragmentShader


class Renderer(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.clearColor = color(0, 0, 0)
        self.glClear()

        # White
        self.currentColor = color(1, 1, 1)
        self.color = color(1, 1, 1)

        self.objects = []

        self.vertexShader = None

        self.primitiveType = TRIANGLES
        self.vertexBuffer = []

        self.activeTexture = None

        self.activeModelMatrix = None

        self.glViewport(0, 0, self.width, self.height)
        self.glCamMatrix()
        self.glProjectionMatrix()

        self.directionalLight = (0, 1, 0)

    def glAddVertices(self, vertices):
        for vert in vertices:
            self.vertexBuffer.append(vert)

    def glPrimitiveAssembly(self, tVerts, tTexCoords, tNormals):
        primitives = []

        if self.primitiveType == TRIANGLES:
            for i in range(0, len(tVerts), 3):
                triangle = []
                # Verts
                verts = []
                verts.append(tVerts[i])
                verts.append(tVerts[i + 1])
                verts.append(tVerts[i + 2])

                # TexCoords
                texCoords = []
                texCoords.append(tTexCoords[i])
                texCoords.append(tTexCoords[i + 1])
                texCoords.append(tTexCoords[i + 2])

                # Normals
                normals = []
                normals.append(tNormals[i])
                normals.append(tNormals[i + 1])
                normals.append(tNormals[i + 2])

                triangle = [verts, texCoords, normals]

                primitives.append(triangle)

        return primitives

    def glClearColor(self, r, g, b):
        self.clearColor = color(r, g, b)

    def glColor(self, r, g, b):
        self.currentColor = color(r, g, b)

    def glPoint(self, x, y, clr=None):
        if (0 <= x < self.width) and (0 <= y < self.height):
            self.pixels[x][y] = clr or self.currentColor

    def glClear(self):
        self.pixels = [
            [self.clearColor for y in range(self.height)] for x in range(self.width)
        ]

        self.zbuffer = [
            [float("inf") for y in range(self.height)] for x in range(self.width)
        ]

    def glTriangle(self, A, B, C, clr=None):
        if A[1] < B[1]:
            A, B = B, A

        if A[1] < C[1]:
            A, C = C, A

        if B[1] < C[1]:
            B, C = C, B

        self.glLine(A, B, clr or self.currentColor)
        self.glLine(B, C, clr or self.currentColor)
        self.glLine(C, A, clr or self.currentColor)

        def flatTop(vA, vB, vC):
            try:
                mCA = (vC[0] - vA[0]) / (vC[1] - vA[1])
                mCB = (vC[0] - vB[0]) / (vC[1] - vB[1])
            except:
                pass
            else:
                x0 = vA[0]
                x1 = vB[0]

                for y in range(int(vA[1]), int(vC[1]), -1):
                    self.glLine(V2(x0, y), V2(x1, y))

                    x0 -= mCA
                    x1 -= mCB

        def flatBottom(vA, vB, vC):
            try:
                mBA = (vB[0] - vA[0]) / (vB[1] - vA[1])
                mCA = (vC[0] - vA[0]) / (vC[1] - vA[1])
            except:
                pass
            else:
                x0 = vB[0]
                x1 = vC[0]

                for y in range(int(vB[1]), int(vA[1])):
                    self.glLine(V2(x0, y), V2(x1, y))

                    x0 += mBA
                    x1 += mCA

        if B[1] == C[1]:
            flatBottom(A, B, C)
        elif A[1] == B[1]:
            flatTop(A, B, C)
        else:
            D = (A[0] + ((B[1] - A[1]) / (C[1] - A[1])) * (C[0] - A[0]), B[1])

            flatBottom(A, B, D)

            flatTop(B, D, C)

    def glTriangle_bc(self, verts, texCoords, normals):
        A = verts[0]
        B = verts[1]
        C = verts[2]

        minX = round(min(A[0], B[0], C[0]))
        maxX = round(max(A[0], B[0], C[0]))
        minY = round(min(A[1], B[1], C[1]))
        maxY = round(max(A[1], B[1], C[1]))

        for x in range(minX, maxX + 1):
            for y in range(minY, maxY + 1):
                if 0 <= x < self.width and 0 <= y < self.height:
                    P = (x, y)
                    bCoords = barycentricCoords(A, B, C, P)

                    if bCoords != None:
                        u, v, w = bCoords

                        z = u * A[2] + v * B[2] + w * C[2]

                        if z < self.zbuffer[x][y]:
                            self.zbuffer[x][y] = z

                            # OJO PARA CREAR SHADERS
                            if self.fragmentShader != None:
                                colorP = self.fragmentShader(
                                    texture=self.activeTexture,
                                    texCoords=texCoords,
                                    normals=normals,
                                    dLight=self.directionalLight,
                                    bCoords=bCoords,
                                    camMatrix=self.camMatrix,
                                    modelMatrix=self.activeModelMatrix,
                                )

                                self.glPoint(
                                    x, y, color(colorP[0], colorP[1], colorP[2])
                                )
                            else:
                                self.glPoint(x, y, colorP)

    def glViewport(self, x, y, width, height):
        self.vpX = x
        self.vpY = y
        self.vpWidth = width
        self.vpHeight = height

        self.vpMatrix = np.matrix(
            [
                [self.vpWidth / 2, 0, 0, self.vpX + self.vpWidth / 2],
                [0, self.vpHeight / 2, 0, self.vpY + self.vpHeight / 2],
                [0, 0, 0.5, 0.5],
                [0, 0, 0, 1],
            ]
        )

    def glCamMatrix(self, translate=(0, 0, 0), rotate=(0, 0, 0)):

        self.camMatrix = self.glModelMatrix(translate, rotate)

        self.viewMatrix = np.linalg.inv(self.camMatrix)

    def glLookAt(self, camPos=(0, 0, 0), eyePos=(0, 0, 0)):
        worldUp = (0, 1, 0)

        forward = np.subtract(camPos, eyePos)
        forward = forward / np.linalg.norm(forward)

        right = np.cross(worldUp, forward)
        right = right / np.linalg.norm(right)

        up = np.cross(forward, right)
        up = up / np.linalg.norm(up)

        self.camMatrix = np.matrix(
            [
                [right[0], up[0], forward[0], camPos[0]],
                [right[1], up[1], forward[1], camPos[1]],
                [right[2], up[2], forward[2], camPos[2]],
                [0, 0, 0, 1],
            ]
        )

        self.viewMatrix = np.linalg.inv(self.camMatrix)

    def glProjectionMatrix(self, fov=60, n=0.1, f=1000):
        aspectRatio = self.vpWidth / self.vpHeight

        t = tan((fov * pi / 180) / 2) * n
        r = t * aspectRatio
        self.projectionMatrix = np.matrix(
            [
                [n / r, 0, 0, 0],
                [0, n / t, 0, 0],
                [0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)],
                [0, 0, -1, 0],
            ]
        )

    def glModelMatrix(self, translate=(0, 0, 0), rotate=(0, 0, 0), scale=(1, 1, 1)):
        translation = np.matrix(
            [
                [1, 0, 0, translate[0]],
                [0, 1, 0, translate[1]],
                [0, 0, 1, translate[2]],
                [0, 0, 0, 1],
            ]
        )

        rolMat = self.glRotationMatrix(rotate[0], rotate[1], rotate[2])

        scaleMat = np.matrix(
            [
                [scale[0], 0, 0, 0],
                [0, scale[1], 0, 0],
                [0, 0, scale[2], 0],
                [0, 0, 0, 1],
            ]
        )

        return translation * rolMat * scaleMat

    def glRotationMatrix(self, pitch=0, yaw=0, roll=0):
        pitch *= pi / 180
        yaw *= pi / 180
        roll *= pi / 180

        pitchMat = np.matrix(
            [
                [1, 0, 0, 0],
                [0, cos(pitch), -sin(pitch), 0],
                [0, sin(pitch), cos(pitch), 0],
                [0, 0, 0, 1],
            ]
        )

        yawMat = np.matrix(
            [
                [cos(yaw), 0, sin(yaw), 0],
                [0, 1, 0, 0],
                [-sin(yaw), 0, cos(yaw), 0],
                [0, 0, 0, 1],
            ]
        )

        rollMat = np.matrix(
            [
                [cos(roll), -sin(roll), 0, 0],
                [sin(roll), cos(roll), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        return pitchMat * yawMat * rollMat

    def glLine(self, v0, v1, clr=None):

        x0 = int(v0[0])
        x1 = int(v1[0])
        y0 = int(v0[1])
        y1 = int(v1[1])

        if x0 == x1 and y0 == y1:
            self.glPoint(x0, y0)
            return

        dy = abs(y1 - y0)
        dx = abs(x1 - x0)

        steep = dy > dx

        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1

        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        dy = abs(y1 - y0)
        dx = abs(x1 - x0)

        offset = 0
        limit = 0.5

        m = dy / dx
        y = y0

        for x in range(x0, x1 + 1):
            if steep:
                # Vertical
                self.glPoint(y, x, clr or self.currentColor)
            else:
                # Horizontal
                self.glPoint(x, y, clr or self.currentColor)

            offset += m

            if offset >= limit:
                if y0 < y1:
                    y += 1
                else:
                    y -= 1

                limit += 1

    def glAddModel(self, model):
        self.objects.append(model)

    def glRender(self):
        transformedVerts = []
        textureCoords = []
        normals = []

        for model in self.objects:
            transformedVerts = []
            textureCoords = []
            normals = []

            self.vertexShader = model.vertexShader
            self.fragmentShader = model.fragmentShader
            self.activeTexture = model.texture

            self.activeModelMatrix = self.glModelMatrix(
                model.translate, model.rotate, model.scale
            )

            for face in model.faces:
                vertCount = len(face)

                v0 = model.vertices[face[0][0] - 1]
                v1 = model.vertices[face[1][0] - 1]
                v2 = model.vertices[face[2][0] - 1]

                if vertCount == 4:
                    v3 = model.vertices[face[3][0] - 1]

                vt0 = model.texcoords[face[0][1] - 1]
                vt1 = model.texcoords[face[1][1] - 1]
                vt2 = model.texcoords[face[2][1] - 1]

                if vertCount == 4:
                    vt3 = model.texcoords[face[3][1] - 1]

                vn0 = model.normals[face[0][2] - 1]
                vn1 = model.normals[face[1][2] - 1]
                vn2 = model.normals[face[2][2] - 1]

                if vertCount == 4:
                    vn3 = model.normals[face[3][2] - 1]

                if self.vertexShader:
                    v0 = self.vertexShader(
                        v0,
                        modelMatrix=self.activeModelMatrix,
                        viewMatrix=self.viewMatrix,
                        projectionMatrix=self.projectionMatrix,
                        vpMatrix=self.vpMatrix,
                        normal=vn0,
                    )
                    v1 = self.vertexShader(
                        v1,
                        modelMatrix=self.activeModelMatrix,
                        viewMatrix=self.viewMatrix,
                        projectionMatrix=self.projectionMatrix,
                        vpMatrix=self.vpMatrix,
                        normal=vn1,
                    )
                    v2 = self.vertexShader(
                        v2,
                        modelMatrix=self.activeModelMatrix,
                        viewMatrix=self.viewMatrix,
                        projectionMatrix=self.projectionMatrix,
                        vpMatrix=self.vpMatrix,
                        normal=vn2,
                    )

                    if vertCount == 4:
                        v3 = self.vertexShader(
                            v3,
                            modelMatrix=self.activeModelMatrix,
                            viewMatrix=self.viewMatrix,
                            projectionMatrix=self.projectionMatrix,
                            vpMatrix=self.vpMatrix,
                            normal=vn3,
                        )

                transformedVerts.append(v0)
                transformedVerts.append(v1)
                transformedVerts.append(v2)

                if vertCount == 4:
                    transformedVerts.append(v0)
                    transformedVerts.append(v2)
                    transformedVerts.append(v3)

                vt0 = model.texcoords[face[0][1] - 1]
                vt1 = model.texcoords[face[1][1] - 1]
                vt2 = model.texcoords[face[2][1] - 1]

                if vertCount == 4:
                    vt3 = model.texcoords[face[3][1] - 1]

                textureCoords.append(vt0)
                textureCoords.append(vt1)
                textureCoords.append(vt2)
                if vertCount == 4:
                    textureCoords.append(vt0)
                    textureCoords.append(vt2)
                    textureCoords.append(vt3)

                vn0 = model.normals[face[0][2] - 1]
                vn1 = model.normals[face[1][2] - 1]
                vn2 = model.normals[face[2][2] - 1]

                if vertCount == 4:
                    vn3 = model.normals[face[3][2] - 1]

                normals.append(vn0)
                normals.append(vn1)
                normals.append(vn2)

                if vertCount == 4:
                    normals.append(vn0)
                    normals.append(vn2)
                    normals.append(vn3)

            primitives = self.glPrimitiveAssembly(
                transformedVerts, textureCoords, normals
            )

            for prim in primitives:
                if self.primitiveType == TRIANGLES:
                    self.glTriangle_bc(prim[0], prim[1], prim[2])

    def glFinish(self, fileName):
        with open(fileName, "wb") as file:
            # Header
            file.write(bytes("B".encode("ascii")))
            file.write(bytes("M".encode("ascii")))
            file.write(dword(14 + 40 + self.width * self.height * 3))
            file.write(dword(0))
            file.write(dword(14 + 40))

            # Info header
            file.write(dword(40))
            file.write(dword(self.width))
            file.write(dword(self.height))
            file.write(word(1))
            file.write(word(24))
            file.write(dword(0))
            file.write(dword(self.width * self.height * 3))
            file.write(dword(0))
            file.write(dword(0))
            file.write(dword(0))
            file.write(dword(0))

            # Color table
            for y in range(self.height):
                for x in range(self.width):
                    file.write(self.pixels[x][y])
