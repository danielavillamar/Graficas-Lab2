from texture import Texture


class Obj(object):
    def __init__(self, fileName):
        with open(fileName, "r") as file:
            self.lines = file.read().splitlines()

        self.vertices = []
        self.texcoords = []
        self.normals = []
        self.faces = []

        for line in self.lines:
            try:
                prefix, value = line.split(" ", 1)
                prefix = prefix.strip()
                value = value.strip()
            except:
                continue

            if prefix == "v":  # Vertices
                self.vertices.append(list(map(float, value.split(" "))))
            elif prefix == "vt":  # Texture Coordinates
                self.texcoords.append(list(map(float, value.split(" "))))
            elif prefix == "vn":  # Normals
                self.normals.append(list(map(float, value.split(" "))))
            elif prefix == "f":  # Faces
                self.faces.append(
                    [list(map(int, vert.split("/"))) for vert in value.split(" ")]
                )
