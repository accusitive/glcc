// OpenGL 3.3 core triangle via SDL2, compiled with glcpp.
//
// Build: make (see Makefile; glcpp + sdl2-config, then make run)
//
// Deliberately does NOT include <GL/gl.h> / <SDL_opengl.h>. Those headers are
// big and full of platform macros; instead we declare the handful of types and
// constants we need and load every GL entry point through
// SDL_GL_GetProcAddress. The only things the program links against are SDL2
// itself (dynamically) -- no libGL at link time. That also makes the X11/GLX
// fallback trivial: swap SDL_GL_GetProcAddress for glXGetProcAddress.

#include <SDL2/SDL.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>

namespace {

// ------------------------------------------------- minimal GL declarations --

using GLenum = unsigned int;
using GLuint = unsigned int;
using GLint = int;
using GLsizei = int;
using GLbitfield = unsigned int;
using GLboolean = unsigned char;
using GLfloat = float;
using GLchar = char;
using GLsizeiptr = long;  // ptrdiff_t-sized

constexpr GLenum GL_COLOR_BUFFER_BIT = 0x4000;
constexpr GLenum GL_TRIANGLES = 0x0004;
constexpr GLenum GL_FLOAT = 0x1406;
constexpr GLenum GL_ARRAY_BUFFER = 0x8892;
constexpr GLenum GL_STATIC_DRAW = 0x88E4;
constexpr GLenum GL_FRAGMENT_SHADER = 0x8B30;
constexpr GLenum GL_VERTEX_SHADER = 0x8B31;
constexpr GLenum GL_COMPILE_STATUS = 0x8B81;
constexpr GLenum GL_LINK_STATUS = 0x8B82;
constexpr GLboolean GL_FALSE = 0;

struct GL {
    void (*ClearColor)(GLfloat, GLfloat, GLfloat, GLfloat);
    void (*Clear)(GLbitfield);
    void (*Viewport)(GLint, GLint, GLsizei, GLsizei);
    void (*GenVertexArrays)(GLsizei, GLuint*);
    void (*BindVertexArray)(GLuint);
    void (*GenBuffers)(GLsizei, GLuint*);
    void (*BindBuffer)(GLenum, GLuint);
    void (*BufferData)(GLenum, GLsizeiptr, const void*, GLenum);
    void (*VertexAttribPointer)(GLuint, GLint, GLenum, GLboolean, GLsizei, const void*);
    void (*EnableVertexAttribArray)(GLuint);
    GLuint (*CreateShader)(GLenum);
    void (*ShaderSource)(GLuint, GLsizei, const GLchar* const*, const GLint*);
    void (*CompileShader)(GLuint);
    void (*GetShaderiv)(GLuint, GLenum, GLint*);
    void (*GetShaderInfoLog)(GLuint, GLsizei, GLsizei*, GLchar*);
    void (*DeleteShader)(GLuint);
    GLuint (*CreateProgram)();
    void (*AttachShader)(GLuint, GLuint);
    void (*LinkProgram)(GLuint);
    void (*GetProgramiv)(GLuint, GLenum, GLint*);
    void (*GetProgramInfoLog)(GLuint, GLsizei, GLsizei*, GLchar*);
    void (*UseProgram)(GLuint);
    void (*DrawArrays)(GLenum, GLint, GLsizei);
};

GL gl;

template <typename Fn>
bool load(Fn& out, const char* name) {
    out = reinterpret_cast<Fn>(SDL_GL_GetProcAddress(name));
    if (!out) std::fprintf(stderr, "missing GL function: %s\n", name);
    return out != nullptr;
}

bool load_gl() {
    bool ok = true;
    ok &= load(gl.ClearColor, "glClearColor");
    ok &= load(gl.Clear, "glClear");
    ok &= load(gl.Viewport, "glViewport");
    ok &= load(gl.GenVertexArrays, "glGenVertexArrays");
    ok &= load(gl.BindVertexArray, "glBindVertexArray");
    ok &= load(gl.GenBuffers, "glGenBuffers");
    ok &= load(gl.BindBuffer, "glBindBuffer");
    ok &= load(gl.BufferData, "glBufferData");
    ok &= load(gl.VertexAttribPointer, "glVertexAttribPointer");
    ok &= load(gl.EnableVertexAttribArray, "glEnableVertexAttribArray");
    ok &= load(gl.CreateShader, "glCreateShader");
    ok &= load(gl.ShaderSource, "glShaderSource");
    ok &= load(gl.CompileShader, "glCompileShader");
    ok &= load(gl.GetShaderiv, "glGetShaderiv");
    ok &= load(gl.GetShaderInfoLog, "glGetShaderInfoLog");
    ok &= load(gl.DeleteShader, "glDeleteShader");
    ok &= load(gl.CreateProgram, "glCreateProgram");
    ok &= load(gl.AttachShader, "glAttachShader");
    ok &= load(gl.LinkProgram, "glLinkProgram");
    ok &= load(gl.GetProgramiv, "glGetProgramiv");
    ok &= load(gl.GetProgramInfoLog, "glGetProgramInfoLog");
    ok &= load(gl.UseProgram, "glUseProgram");
    ok &= load(gl.DrawArrays, "glDrawArrays");
    return ok;
}

// ------------------------------------------------------------ GL helpers ---

const char* const kVertexSrc = R"(#version 330 core
layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec3 a_color;
out vec3 v_color;
void main() {
    v_color = a_color;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
)";

const char* const kFragmentSrc = R"(#version 330 core
in vec3 v_color;
out vec4 frag_color;
void main() {
    frag_color = vec4(v_color, 1.0);
}
)";

GLuint compile_shader(GLenum type, const char* src) {
    const GLuint shader = gl.CreateShader(type);
    gl.ShaderSource(shader, 1, &src, nullptr);
    gl.CompileShader(shader);

    GLint ok = 0;
    gl.GetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        gl.GetShaderInfoLog(shader, sizeof log, nullptr, log);
        std::fprintf(stderr, "shader compile failed:\n%s\n", log);
        return 0;
    }
    return shader;
}

GLuint build_program(const char* vs_src, const char* fs_src) {
    const GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_src);
    const GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_src);
    if (!vs || !fs) return 0;

    const GLuint program = gl.CreateProgram();
    gl.AttachShader(program, vs);
    gl.AttachShader(program, fs);
    gl.LinkProgram(program);
    gl.DeleteShader(vs);
    gl.DeleteShader(fs);

    GLint ok = 0;
    gl.GetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        gl.GetProgramInfoLog(program, sizeof log, nullptr, log);
        std::fprintf(stderr, "program link failed:\n%s\n", log);
        return 0;
    }
    return program;
}

// ----------------------------------------------------------- SDL wrappers --

using WindowPtr = std::unique_ptr<SDL_Window, decltype(&SDL_DestroyWindow)>;
using ContextPtr = std::unique_ptr<void, decltype(&SDL_GL_DeleteContext)>;

struct Vertex {
    GLfloat x, y;
    GLfloat r, g, b;
};

constexpr Vertex kTriangle[3] = {
    {-0.6f, -0.5f, 1.0f, 0.2f, 0.2f},
    { 0.6f, -0.5f, 0.2f, 1.0f, 0.2f},
    { 0.0f,  0.6f, 0.2f, 0.4f, 1.0f},
};

}  // namespace

int main(int, char**) {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
        return EXIT_FAILURE;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

    WindowPtr window(SDL_CreateWindow("triangle", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                      800, 600, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE),
                     SDL_DestroyWindow);
    if (!window) {
        std::fprintf(stderr, "SDL_CreateWindow: %s\n", SDL_GetError());
        SDL_Quit();
        return EXIT_FAILURE;
    }

    ContextPtr context(SDL_GL_CreateContext(window.get()), SDL_GL_DeleteContext);
    if (!context) {
        std::fprintf(stderr, "SDL_GL_CreateContext: %s\n", SDL_GetError());
        SDL_Quit();
        return EXIT_FAILURE;
    }
    SDL_GL_SetSwapInterval(1);  // vsync

    if (!load_gl()) {
        SDL_Quit();
        return EXIT_FAILURE;
    }

    const GLuint program = build_program(kVertexSrc, kFragmentSrc);
    if (!program) {
        SDL_Quit();
        return EXIT_FAILURE;
    }

    GLuint vao = 0, vbo = 0;
    gl.GenVertexArrays(1, &vao);
    gl.BindVertexArray(vao);
    gl.GenBuffers(1, &vbo);
    gl.BindBuffer(GL_ARRAY_BUFFER, vbo);
    gl.BufferData(GL_ARRAY_BUFFER, sizeof kTriangle, kTriangle, GL_STATIC_DRAW);
    gl.VertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                           reinterpret_cast<const void*>(offsetof(Vertex, x)));
    gl.EnableVertexAttribArray(0);
    gl.VertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                           reinterpret_cast<const void*>(offsetof(Vertex, r)));
    gl.EnableVertexAttribArray(1);

    bool running = true;
    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = false;
            if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE) running = false;
        }

        int w = 0, h = 0;
        SDL_GL_GetDrawableSize(window.get(), &w, &h);
        gl.Viewport(0, 0, w, h);

        gl.ClearColor(0.08f, 0.09f, 0.11f, 1.0f);
        gl.Clear(GL_COLOR_BUFFER_BIT);

        gl.UseProgram(program);
        gl.BindVertexArray(vao);
        gl.DrawArrays(GL_TRIANGLES, 0, 3);

        SDL_GL_SwapWindow(window.get());
    }

    // window/context are released by their unique_ptrs before SDL_Quit runs
    // only if destroyed first, so do that explicitly.
    context.reset();
    window.reset();
    SDL_Quit();
    return EXIT_SUCCESS;
}
