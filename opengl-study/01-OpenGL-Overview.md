# OpenGL Overview

> see: https://www.songho.ca/opengl/gl_overview.html

-   [Introduction](#opengl-introduction)
-   [State Machine](#state-machine)
-   [glBegin() and glEnd()](#glbegin-and-glend)
-   [glFlush() and glFinish()](#glflush-and-glfinish)

## OpenGL Introduction

OpenGL is a software interface to graphics hardware. It is designed as a hardware-independent interface to be used for many different hardware platforms. OpenGL programs can also work across a network (client-server paradigm) even if the client and server are different kinds of computers. The client in OpenGL is a computer on which an OpenGL program actually executes, and the server is a computer that performs the drawings.  
  
OpenGL uses the prefix  _**gl**_  for core OpenGL commands and  _**glu**_  for commands in OpenGL Utility Library. Similarly, OpenGL constants begin with  _**GL_**_  and use all capital letters. OpenGL also uses suffix to specify the number of arguments and data type passed to a OpenGL call.

```cpp
glColor3f(1, 0, 0);         // set rendering color to red with 3 floating numbers
glColor4d(0, 1, 0, 0.2);    // set color to green with 20% of opacity (double)
glVertex3fv(vertex);        // set x-y-z coordinates using pointer
```

## State Machine

OpenGL is a state machine. Modes and attributes in OpenGL will be remained in effect until they are changed. Most state variables can be enabled or disabled with  **glEnable()**  or  **glDisable()**. You can also check if a state is currently enabled or disabled with  **glIsEnabled()**. You can save or restore a collection of state variables into/from attribute stacks using  **glPushAttrib()**  or  **glPopAttrib()**.  **GL_ALL_ATTRIB_BITS**  parameter can be used to save/restore all states. The number of stacks must be at least 16 in OpenGL standard.  
_(Check your maximum stack size with  [glinfo](https://www.songho.ca/opengl/files/glinfo.zip).)_

```cpp
glPushAttrib(GL_LIGHTING_BIT);    // elegant way to change states because
    glDisable(GL_LIGHTING);       // you can restore exact previous states
    glEnable(GL_COLOR_MATERIAL);  // after calling glPopAttrib()
glPushAttrib(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DITHER);
    glEnable(GL_BLEND);

... // do something

glPopAttrib();                    // restore GL_COLOR_BUFFER_BIT
glPopAttrib();                    // restore GL_LIGHTING_BIT
```

## glBegin and glEnd()

In order to draw geometric primitives (points, lines, triangles, etc) in OpenGL, you can specify a list of vertex data between glBegin() and glEnd(). This method is called immediate mode. (You may draw geometric primitives using other methods such as  [vertex array](https://www.songho.ca/opengl/gl_vertexarray.html).)

```cpp
glBegin(GL_TRIANGLES);
    glColor3f(1, 0, 0);     // set vertex color to red
    glVertex3fv(v1);        // draw a triangle with v1, v2, v3
    glVertex3fv(v2);
    glVertex3fv(v3);
glEnd();
```

There are 10 types of primitives in OpenGL; GL_POINTS, GL_LINES, GL_LINE_STRIP, GL_LINE_LOOP, GL_TRIANGLES, GL_TRIANGLE_STRIP, GL_TRIANGLE_FAN, GL_QUADS, GL_QUAD_STRIP, and GL_POLYGON.

Note that not all of OpenGL commands can be placed in between glBegin() and glEnd(). Only a subset of commands can be used; glVertex*(), glColor*(), glNormal*(), glTexCoord*(), glMaterial*(), glCallList(), etc.


## glFlush() and glFinish()

Similar to computer IO buffer, OpenGL commands are not executed immediately. All commands are stored in buffers first, including network buffers and the graphics accelerator itself, and are awaiting execution until buffers are full. For example, if an application runs over the network, it is much more efficient to send a collection of commands in a single packet than to send each command over network one at a time.

**glFlush()**  empties all commands in these buffers and forces all pending commands to be executed immediately without waiting buffers are full. Therefore  **glFlush()**  guarantees that all OpenGL commands made up to that point will complete executions in a finite amount time after calling  **glFlush()**. And  **glFlush()**  does not wait until previous executions are complete and may return immediately to your program. So you are free to send more commands even though previously issued commands are not finished.

**glFinish()**  flushes buffers and forces commands to begin execution as  **glFlush()**  does, but  **glFinish()**  blocks other OpenGL commands and waits for all execution is complete. Consequently,  **glFinish()**  does not return to your program until all previously called commands are complete. It might be used to `synchronize` tasks or to measure exact elapsed time that certain OpenGL commands are executed.