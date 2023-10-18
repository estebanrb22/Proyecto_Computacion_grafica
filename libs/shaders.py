from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
from PIL import Image

from libs.gpu_shape import GPUShape

SIZE_IN_BYTES = 4


def textureSimpleSetup(imgName, sWrapMode, tWrapMode, minFilterMode, maxFilterMode):
     # wrapMode: GL_REPEAT, GL_CLAMP_TO_EDGE
     # filterMode: GL_LINEAR, GL_NEAREST
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)

    # texture wrapping params
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, sWrapMode)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, tWrapMode)

    # texture filtering params
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilterMode)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, maxFilterMode)

    image = Image.open(imgName)
    img_data = np.array(image, np.uint8)

    if image.mode == "RGB":
        internalFormat = GL_RGB
        format = GL_RGB
    elif image.mode == "RGBA":
        internalFormat = GL_RGBA
        format = GL_RGBA
    else:
        print("Image mode not supported.")
        raise Exception()

    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, image.size[0], image.size[1], 0, format, GL_UNSIGNED_BYTE, img_data)

    return texture


class SimpleModelViewProjectionShaderProgram:

    def __init__(self):

        vertex_shader = """
            #version 330
            in vec3 position;
            in vec3 color;
            in vec3 normal;
            flat out vec4 vertexColor;
            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;

            void main()
            {
                vec3 vertexPos = vec3(model * vec4(position, 1.0));
                gl_Position = projection * view * vec4(vertexPos, 1.0);
                vertexColor = vec4(color, 1.0);

                vec3 normals = normal; // No hace nada
            }
            """

        fragment_shader = """
            #version 330
            flat in vec4 vertexColor;
            out vec4 fragColor;
            void main()
            {
                fragColor = vertexColor;
            }
            """

        # Binding artificial vertex array object for validation
        VAO = glGenVertexArrays(1)
        glBindVertexArray(VAO)


        self.shaderProgram = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(vertex_shader, OpenGL.GL.GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(fragment_shader, OpenGL.GL.GL_FRAGMENT_SHADER))


    def setupVAO(self, gpuShape):

        glBindVertexArray(gpuShape.vao)

        glBindBuffer(GL_ARRAY_BUFFER, gpuShape.vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpuShape.ebo)

        # 3d vertices + rgb color + 3d normals => 3*4 + 3*4 + 3*4 = 36 bytes
        position = glGetAttribLocation(self.shaderProgram, "position")
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 36, ctypes.c_void_p(0))
        glEnableVertexAttribArray(position)

        color = glGetAttribLocation(self.shaderProgram, "color")
        glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, 36, ctypes.c_void_p(12))
        glEnableVertexAttribArray(color)

        normal = glGetAttribLocation(self.shaderProgram, "normal")
        if normal >=0:
            glVertexAttribPointer(normal, 3, GL_FLOAT, GL_FALSE, 36, ctypes.c_void_p(24))
            glEnableVertexAttribArray(normal)

        # Unbinding current vao
        glBindVertexArray(0)


    def drawCall(self, gpuShape, mode=GL_TRIANGLES):
        assert isinstance(gpuShape, GPUShape)

        # Binding the VAO and executing the draw call
        glBindVertexArray(gpuShape.vao)

        glDrawElements(mode, gpuShape.size, GL_UNSIGNED_INT, None)

        # Unbind the current VAO
        glBindVertexArray(0)


class SimpleTextureModelViewProjectionShaderProgram:

    def __init__(self, vertexshader, fragmentshader):

        vertex_shader = """
            #version 330
            in vec3 position;
            in vec2 texCoords;
            in vec3 normal;
            out vec2 fragTexCoords;
            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;

            void main()
            {
                vec3 vertexPos = vec3(model * vec4(position, 1.0));
                gl_Position = projection * view * vec4(vertexPos, 1.0);
                fragTexCoords = vec2(texCoords[0],texCoords[1]);

                vec3 normals = normal; // No hace nada
            }
            """

        fragment_shader = """
            #version 330
            in vec2 fragTexCoords;
            out vec4 fragColor;
            uniform sampler2D samplerTex;
            void main()
            {
                vec4 textureColor = texture(samplerTex, fragTexCoords);
                fragColor = textureColor;
            }
            """

        # Binding artificial vertex array object for validation
        VAO = glGenVertexArrays(1)
        glBindVertexArray(VAO)


        self.shaderProgram = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(vertexshader, OpenGL.GL.GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(fragmentshader, OpenGL.GL.GL_FRAGMENT_SHADER))


    def setupVAO(self, gpuShape):

        glBindVertexArray(gpuShape.vao)

        glBindBuffer(GL_ARRAY_BUFFER, gpuShape.vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpuShape.ebo)

        # 3d vertices + rgb color + 3d normals => 3*4 + 2*4 + 3*4 = 32 bytes
        position = glGetAttribLocation(self.shaderProgram, "position")
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        glEnableVertexAttribArray(position)

        color = glGetAttribLocation(self.shaderProgram, "texCoords")
        glVertexAttribPointer(color, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        glEnableVertexAttribArray(color)

        normal = glGetAttribLocation(self.shaderProgram, "normal")
        if normal >=0:
            glVertexAttribPointer(normal, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))
            glEnableVertexAttribArray(normal)

        # Unbinding current vao
        glBindVertexArray(0)


    def drawCall(self, gpuShape, mode=GL_TRIANGLES):
        assert isinstance(gpuShape, GPUShape)

        # Binding the VAO and executing the draw call
        glBindVertexArray(gpuShape.vao)
        glBindTexture(GL_TEXTURE_2D, gpuShape.texture)

        glDrawElements(mode, gpuShape.size, GL_UNSIGNED_INT, None)

        # Unbind the current VAO
        glBindVertexArray(0)


class MultipleLightTexturePhongShaderProgram:

    def __init__(self):
        
        vertex_shader = """
        #version 330 core
         
        in vec3 position;
        in vec2 texCoords;
        in vec3 normal;

        out vec3 fragPosition;
        out vec2 fragTexCoords;
        out vec3 fragNormal;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform mat4 transform;

        void main(){

            fragPosition = vec3(transform * model * vec4(position, 1.0));
            fragTexCoords = vec2(texCoords[0], 1 - texCoords[1]);
            fragNormal = mat3(transpose(inverse(transform * model))) * normal;  
            gl_Position = projection * view * vec4(fragPosition, 1.0);

        }
        """
        
        fragment_shader = """
        #version 330 core

        in vec3 fragNormal;
        in vec3 fragPosition;
        in vec2 fragTexCoords;

        out vec4 fragColor;

        struct Material {
            vec3 ambient;
            vec3 diffuse;
            vec3 specular;
            float shininess;
        }; 

        struct PointLight {
            vec3 position;
            
            float constant;
            float linear;
            float quadratic;
            
            vec3 ambient;
            vec3 diffuse;
            vec3 specular;
        };

        uniform vec3 viewPosition; 
        #define NR_POINT_LIGHTS 1

        uniform PointLight pointLights[NR_POINT_LIGHTS];
        uniform Material material;

        uniform sampler2D samplerTex;

        vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir);

        void main(){
            vec3 norm = normalize(fragNormal);
            vec3 viewDir = normalize(viewPosition - fragPosition);

            vec3 result = vec3(0.0,0.0,0.0);

            for(int i = 0; i < NR_POINT_LIGHTS; i++)
                result += CalcPointLight(pointLights[i], norm, fragPosition, viewDir);

            vec4 fragOriginalColor = texture(samplerTex, fragTexCoords);
            vec3 resultFinal = result * fragOriginalColor.rgb;

            fragColor = vec4(resultFinal, 1.0);

        }

        // calculates the color when using a point light.
        vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir)
        {
            vec3 lightDir = normalize(light.position - fragPos);
            // diffuse shading
            float diff = max(dot(normal, lightDir), 0.0);
            // specular shading
            vec3 reflectDir = reflect(-lightDir, normal);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
            // attenuation
            float distance = length(light.position - fragPos);
            float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));    
            // combine results
            vec3 ambient = light.ambient;
            vec3 diffuse = light.diffuse * diff;
            vec3 specular = light.specular * spec;

            //ambient *= attenuation;
            diffuse *= attenuation;
            specular *= attenuation;
            return (ambient + diffuse + specular);
        }
        """
        
        # Binding artificial vertex array object for validation
        VAO = glGenVertexArrays(1)
        glBindVertexArray(VAO)


        self.shaderProgram = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(vertex_shader, OpenGL.GL.GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(fragment_shader, OpenGL.GL.GL_FRAGMENT_SHADER))


    def setupVAO(self, gpuShape):

        glBindVertexArray(gpuShape.vao)

        glBindBuffer(GL_ARRAY_BUFFER, gpuShape.vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpuShape.ebo)

        # 3d vertices + rgb color + 3d normals => 3*4 + 2*4 + 3*4 = 32 bytes
        position = glGetAttribLocation(self.shaderProgram, "position")
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        glEnableVertexAttribArray(position)
        
        color = glGetAttribLocation(self.shaderProgram, "texCoords")
        glVertexAttribPointer(color, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        glEnableVertexAttribArray(color)

        normal = glGetAttribLocation(self.shaderProgram, "normal")
        glVertexAttribPointer(normal, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))
        glEnableVertexAttribArray(normal)

        # Unbinding current vao
        glBindVertexArray(0)


    def drawCall(self, gpuShape, mode=GL_TRIANGLES):
        assert isinstance(gpuShape, GPUShape)

        # Binding the VAO and executing the draw call
        glBindVertexArray(gpuShape.vao)
        glBindTexture(GL_TEXTURE_2D, gpuShape.texture)

        glDrawElements(mode, gpuShape.size, GL_UNSIGNED_INT, None)

        # Unbind the current VAO
        glBindVertexArray(0)

class PointLight:
    def __init__(self, pos: list, ambient, diffuse, specular, constant, linear, quadratic) -> None:
        self.attribs = {
            "position": np.array(pos),
            "ambient": np.array(ambient),
            "diffuse": np.array(diffuse),
            "specular": np.array(specular),
            "constant": constant,
            "linear": linear,
            "quadratic": quadratic
        }

    def set_pos(self, x, y, z):
        self.attribs["position"] = np.array([x, y, z])

    def set_ambient(self, r, g, b):
        self.attribs["ambient"] = np.array([r, g, b])

    def set_diffuse(self, r, g, b):
        self.attribs["diffuse"] = np.array([r, g, b])

    def set_specular(self, r, g, b):
        self.attribs["specular"] = np.array([r, g, b])

    def show_colors(self):
        colors = f"ambient: ({self.attribs['ambient']})\n"
        colors += f"diffuse: ({self.attribs['diffuse']})\n"
        colors += f"specular: ({self.attribs['specular']})\n"
        return colors
    
def update_lights(pointlights, pipeline, amb_scale, diff_scale, spec_scale):
    for i, pointlight in enumerate(pointlights):
        light_color = np.array([1.0, 0.989, 0.9], dtype = float)
        pointlight.set_ambient(*(light_color * amb_scale))
        pointlight.set_diffuse(*(light_color * diff_scale))
        pointlight.set_specular(*(light_color * spec_scale))
        for key in pointlight.attribs.keys():
            attrib = pointlight.attribs[key]
            if isinstance(attrib, float):
                glUniform1f(glGetUniformLocation(pipeline.shaderProgram, f"pointLights[{i}].{key}"), attrib)
            elif isinstance(attrib, np.ndarray):
                glUniform3f(glGetUniformLocation(pipeline.shaderProgram, f"pointLights[{i}].{key}"), *attrib)