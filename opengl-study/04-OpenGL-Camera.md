# 04 - OpenGL Camera

@(3D Computer Vision)[OpenGL]

> see https://www.songho.ca/opengl/gl_camera.html

[TOC]


**Related Topics:**  [OpenGL Transform](https://www.songho.ca/opengl/gl_transform.html),  [OpenGL Projection Matrix](https://www.songho.ca/opengl/gl_projectionmatrix.html),  [Quaternion to Rotation Matrix](https://www.songho.ca/opengl/gl_quaternion.html)  
**Download:**  [OrbitCamera.zip](https://www.songho.ca/opengl/files/OrbitCamera.zip),  [trackball.zip](https://www.songho.ca/opengl/files/trackball.zip)

-   [Overview](#overview)
-   [Camera LookAt](#lookat)
-   [Camera Rotation (Pitch, Yaw, Roll)](#rotate)
-   [Camera Shifting](#shift)
-   [Camera Forwarding](#forward)
-   [Example: Orbit Camera](#example1)
-   [Example: Trackball](#example2)


## Overview

![OpenGL Eye Space | center ](./data/gl_camera01.png)  

OpenGL camera is always at origin and facing to $-Z$ in eye space.

OpenGL doesn't explicitly define a camera object or a specific matrix for camera transformation. Instead, OpenGL transforms the entire scene (_including the camera_) inversely to a space, where a fixed camera is at the origin (0,0,0) and always looking along -Z axis. This space is called  **eye space**.

Because of this, OpenGL uses a single GL_MODELVIEW matrix for both object transformation to world space and camera (view) transformation to eye space.

You may break it down into 2 logical sub matrices:

$$ M_{\text{modelView}} = M_{view} \cdot M_{model} $$

That is, each object in a scene is transformed with its own  $M_{model} $  first, then the entire scene is transformed reversely with  $M_\text{view}$. In this page, we will discuss only  $M_\text{view}$  for camera transformation in OpenGL.

## LookAt

[gluLookAt()](http://www.opengl.org/sdk/docs/man2/xhtml/gluLookAt.xml)  is used to construct a viewing matrix where a camera is located at the eye position $(x_e, y_e, z_e)$ and looking at (or rotating to) the target point $(x_t, y_t, z_t)$. The eye position and target are defined in world space. This section describes how to implement the viewing matrix equivalent to  **gluLookAt()**.

Camera's **lookAt** transformation consists of 2 transformations; translating the whole scene inversely from the eye position to the origin ($M_T$), and then rotating the scene with reverse orientation ($M_R$), so the camera is positioned at the origin and facing to the $-Z$ axis.  

$$
M_{\mathrm{view}}=M_{\mathrm{R}} M_{\mathrm{T}}=\left(\begin{array}{cccc}
r_0 & r_4 & r_8 & 0 \\
r_1 & r_5 & r_9 & 0 \\
r_2 & r_6 & r_{10} & 0 \\
0 & 0 & 0 & 1
\end{array}\right)
\left(\begin{array}{cccc}
1 & 0 & 0 & t_x \\
0 & 1 & 0 & t_y \\
0 & 0 & 1 & t_z \\
0 & 0 & 0 & 1
\end{array}\right)
$$

That is,
$$
M_{\mathrm{view}}=M_{\mathrm{R}} M_{\mathrm{T}}
=\left(\begin{array}{cccc}
r_0 & r_4 & r_8 & r_0 t_x+r_4 t_y+r_8 t_z \\
r_1 & r_5 & r_9 & r_1 t_x+r_5 t_y+r_9 t_z \\
r_2 & r_6 & r_{10} & r_2 t_x+r_6 t_y+r_{10} t_z \\
0 & 0 & 0 & 1
\end{array}\right)
$$

![Camera lookAt animation |center](./data/gl_camera02.gif)  


Suppose a camera is located at `(2, 0, 3)` and looking at `(0, 0, 0)` in world space. In order to construct the viewing matrix for this case, we need to translate the world to `(-2, 0, -3)` and rotate it about -33.7 degrees along Y-axis. As a result, the virtual camera becomes facing to $-Z$ axis at the origin.

The translation part of lookAt is easy. You simply move the camera position to the origin. The translation matrix  $M_T$  would be (replacing the 4th column with the negated eye position):

$$
M_T = \left(\begin{array}{cccc}
1 & 0 & 0 & -x_e \\
0 & 1 & 0 & -y_e  \\
0 & 0 & 1 &-z_e  \\
0 & 0 & 0 & 1
\end{array}\right)
$$

![left, up and forward vector | center | 400x0](./data/gl_camera05.png)  


The rotation part  $M_R$  of lookAt is much harder than translation because you have to calculate 1st, 2nd and 3rd columns of the rotation matrix all together.

- First, compute the normalized forward vector $f$  from the target position  $v_t$ to the eye position $v_e$  of the rotation matrix. Note that the forward vector is from the target to the eye position, not eye to target because the scene is actually rotated, not the camera is.  

$$ v_e - v_t = (x_e - x_t, y_e - y_t,  z_e - z_t ), \text{ then } \Rightarrow f = \frac{v_e - v_t}{|| v_e - v_t ||}$$

- Second, compute the normalized left vector $l$ by performing cross-product with a given camera's up vector. If the up vector is not provided, you may use $(0, 1, 0)$ by assuming the camera is straight up to $+Y$ axis.  
$$ \text{left} = \text{up}  \times \text{f}, \text{ then }   \Rightarrow  l = \frac{left}{|| \text{ left } ||}
$$

- Finally, re-calculate the normalized up vector $u$ by doing cross product the forward $f$ and left $l$ vectors, so all 3 vectors are orthonormal (perpendicular and unit length). Note we do not normalize the up vector because the cross product of 2 perpendicular unit vectors also produces a unit vector: $u = f \times l$.  

These 3 basis vectors,  $i$, $u$ and $f$ are used to construct the rotation matrix  $M_R$  of lookAt, however, the rotation matrix must be inverted. Suppose a camera is located above a scene. The whole scene must rotate downward inversely, so the camera is facing to $-Z$ axis. In a similar way, if the camera is located at the left of the scene, the scene should rotate to the right in order to align the camera to $-Z$ axis. The following diagrams show why the rotation matrix must be inverted.

- The scene rotates downward if the camera is above:

![Camera pitch animation |center](./data/gl_camera03.gif)  

- The scene rotates to the right if the camera is on left:

![Camera yaw animation | center](./data/gl_camera04.gif)  

Therefore, the rotation part  $M_R$  of lookAt is finding the rotation matrix from the target to the eye position, then inverting it. And, the inverse matrix is equal to its transpose matrix because it is orthogonal (i.e., each column has a unit length and is perpendicular to the other column).  
![Ratation matrix of lookAt | center | 0x200](./data/gl_camera_eq11.png)

Finally, the view matrix for the camera's lookAt transform is multiplying  $M_T$  and  $M_R$  together;  
![OpenGL LookAt matrix | center | 0x200 ](./data/gl_camera_eq12.png)

Here is C++ snippet to construct the view matrix for the camera's lookAt transformation. Please see the details in  [OrbitCamera.cpp](https://www.songho.ca/opengl/files/OrbitCamera.zip).

```cpp
// dependency: Vector3 and Matrix4
struct Vector3
{
    float x;
    float y;
    float z;
};

class Matrix4
{
    float m[16];
}

///////////////////////////////////////////////////////////////////////////////
// equivalent to gluLookAt()
// It returns 4x4 matrix
///////////////////////////////////////////////////////////////////////////////
Matrix4 lookAt(Vector3& eye, Vector3& target, Vector3& upDir)
{
    // compute the forward vector from target to eye
    Vector3 forward = eye - target;
    forward.normalize();                 // make unit length

    // compute the left vector
    Vector3 left = upDir.cross(forward); // cross product
    left.normalize();

    // recompute the orthonormal up vector
    Vector3 up = forward.cross(left);    // cross product

    // init 4x4 matrix
    Matrix4 matrix;
    matrix.identity();

    // set rotation part, inverse rotation matrix: M^-1 = M^T for Euclidean transform
    matrix[0] = left.x;
    matrix[4] = left.y;
    matrix[8] = left.z;
    matrix[1] = up.x;
    matrix[5] = up.y;
    matrix[9] = up.z;
    matrix[2] = forward.x;
    matrix[6] = forward.y;
    matrix[10]= forward.z;

    // set translation part
    matrix[12]= -left.x * eye.x - left.y * eye.y - left.z * eye.z;
    matrix[13]= -up.x * eye.x - up.y * eye.y - up.z * eye.z;
    matrix[14]= -forward.x * eye.x - forward.y * eye.y - forward.z * eye.z;

    return matrix;
}
```

### Rotation (Pitch, Yaw, Roll)

![camera pitch, yaw roll | center | 400x0 ](./data/gl_camera07.png)  

Other types of camera's rotations are  **pitch**,  **yaw**, and  **roll**  rotating at the position of the camera.  **Pitch**  is rotating the camera up and down around the camera's local left axis (+X axis).  **Yaw**  is rotating left and right around the camera's local up axis (+Y axis). And,  **Roll**  is rotating it around the camera's local forward axis (+Z axis).

First, you need to move the whole scene inversely from the camera's position to the origin $(0,0,0)$, same as  [lookAt()](#lookat)  function. Then, rotate the scene along the rotation axis; pitch (X), yaw (Y) and roll (Z) independently.

If multiple rotations are applied, combine them sequentially. Note that the different order of rotations produces a different result. A commonly used rotation order is  **roll -> yaw -> pitch**. It produces a free-look camera or first-person shooter camera interface.

Keep in mind the directions of the rotations (right-hand rule); a positive pitch angle is rotating the camera downward, a positive yaw angle means rotating the camera left, and a positive roll angle is rotating the camera clockwise.

### Shifting
To do ...

### Forwarding
To do ...

### Example: OpenGL Orbit Camera

![OpenGL OrbitCamera |center ](./data/gl_orbitCamera01.png)

Download source and binary:  
_(Updated: 2021-03-10)_

[![](https://www.songho.ca/files/windows-folder-icon32.png)OrbitCamera.zip](https://www.songho.ca/opengl/files/OrbitCamera.zip)  _(include VS 2015 project)_

### Example: Trackball

![OpenGL Trackball |center ](./gl_trackball01.png)

Download:  [trackball.zip](https://www.songho.ca/opengl/files/trackball.zip)