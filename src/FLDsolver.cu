// implementation of Flux-Limited Diffusion for Multiple Scattering in Participating Media

#define HELPFUL_BUG 1
// this is the most tricky part of this algorithm
// the epsilons have very huge impact on the diffusion result

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstring>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <helper_cuda.h>
#define KERNEL __global__
#define DEVICE __device__
#define HOST __host__
#else
#define KERNEL
#define DEVICE
#define HOST
#endif
//////////////////////////////////////////////////////////////////////////

#define M_PI 3.14159265358979323846f
#define M_TWO_PI 6.283185307179586f
#define M_ONE_OVER_FOUR_PI 0.079577471545948f
#define M_FOUR_PI 12.566370614359172f
#define M_EXP 2.718281828459046f
#define FLT_MAX 3.402823466e+38f

#define LINEAR_INTERPOLATOR sampleLinearAccurately
// #define LINEAR_INTERPOLATOR sampleNearestAccurately // for debugging and visuals

#ifdef __CUDACC__
#define STEP_SIZE_REFINE 0.5f //0.1f
#else
#define STEP_SIZE_REFINE 2.0f;
#endif

#define MEDIUM_SCALE 2.0f //1.0f // not linear dependent for the image, need to rerun the diffusion process

float randf()
{
    return rand() / (RAND_MAX + 1.0f);
}
HOST DEVICE float clamp(float x)
{
    return x < 0 ? 0 : x > 1 ? 1 : x;
}
HOST DEVICE int toInt(float x)
{
    return int(powf(clamp(x), 1.0f / 2.2f) * 255);
//     return int(clamp(x) * 255);
}
HOST DEVICE float fMin(float a, float b)
{
    return a < b ? a : b;
}
HOST DEVICE float fMax(float a, float b)
{
    return a > b ? a : b;
}
HOST DEVICE int iMin(int a, int b)
{
    return a < b ? a : b;
}
HOST DEVICE int iMax(int a, int b)
{
    return a > b ? a : b;
}

struct integer3
{
    int x, y, z;
};

int divideUp(int a, int b)
{
    return (a + b - 1) / b;
}

struct vec3
{
    union 
    {
        struct
        {
            float x, y, z;
        };
        struct 
        {
            float r, g, b;
        };
        float data[3];
    };

    HOST DEVICE vec3()
        : x(0), y(0), z(0)
    {
    }
    HOST DEVICE vec3(float a)
        : x(a), y(a), z(a)
    {
    }
    HOST DEVICE vec3(float e1, float e2, float e3)
        : x(e1), y(e2), z(e3)
    {
    }
    HOST DEVICE const float& operator[] (int n) const
    {
        return data[n];
    }
    HOST DEVICE float& operator[] (int n)
    {
        return data[n];
    }
};

void debugPrint(vec3 a)
{
    printf("%f, %f, %f\n", a.x, a.y, a.z);
}
void debugPrint(float a)
{
    printf("%f\n", a);
}
void debugPrint(int a)
{
    printf("%d\n", a);
}

namespace vecOperators
{
    HOST DEVICE vec3 operator+ (vec3 a, vec3 b)
    {
        return vec3(a.x + b.x, a.y + b.y, a.z + b.z);
    }
    HOST DEVICE vec3& operator+= (vec3& a, vec3 b)
    {
        a = a + b;
        return a;
    }
    HOST DEVICE vec3 operator- (vec3 a, vec3 b)
    {
        return vec3(a.x - b.x, a.y - b.y, a.z - b.z);
    }
    HOST DEVICE vec3 operator* (vec3 a, vec3 b)
    {
        return vec3(a.x * b.x, a.y * b.y, a.z * b.z);
    }
    HOST DEVICE vec3& operator*= (vec3& a, vec3 b)
    {
        a = a * b;
        return a;
    }
    HOST DEVICE vec3 operator/ (vec3 a, vec3 b)
    {
        return vec3(a.x / b.x, a.y / b.y, a.z / b.z);
    }
    HOST DEVICE vec3 operator- (vec3 a)
    {
        return vec3(-a.x, -a.y, -a.z);
    }
    HOST DEVICE float length(vec3 a)
    {
        return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
    }
    HOST DEVICE float lengthSquare(vec3 a)
    {
        return a.x * a.x + a.y * a.y + a.z * a.z;
    }
    HOST DEVICE vec3 normalize(vec3 a)
    {
        return a / length(a);
    }
    HOST DEVICE vec3 sqrt(vec3 a)
    {
        return vec3(sqrtf(a.x), sqrtf(a.y), sqrtf(a.z));
    }
    HOST DEVICE vec3 exp(vec3 a)
    {
        return vec3(expf(a.x), expf(a.y), expf(a.z));
//         return vec3(powf(M_EXP, a.x), powf(M_EXP, a.y), powf(M_EXP, a.z));
    }
    HOST DEVICE vec3 pow(vec3 a, float e)
    {
        return vec3(powf(a.x, e), powf(a.y, e), powf(a.z, e));
    }
    HOST DEVICE vec3 vMin(vec3 a, vec3 b)
    {
        return vec3(fMin(a.x, b.x), fMin(a.y, b.y), fMin(a.z, b.z));
    }
    HOST DEVICE vec3 vMax(vec3 a, vec3 b)
    {
        return vec3(fMax(a.x, b.x), fMax(a.y ,b.y), fMax(a.z, b.z));
    }
    HOST DEVICE float minElementOf(vec3 a)
    {
        return fMin(fMin(a.x, a.y), a.z);
    }
    HOST DEVICE float maxElementOf(vec3 a)
    {
        return fMax(fMax(a.x, a.y), a.z);
    }
}
using namespace vecOperators;

struct ray
{
    vec3 origin;
    vec3 direction;

    HOST DEVICE ray(vec3 o, vec3 d)
        : origin(o), direction(d)
    {
    }
    HOST DEVICE vec3 proceed(float t)
    {
        return origin + direction * t;
    }
};

DEVICE bool CUDASDK_intersectBox(ray r, vec3 boxmin, vec3 boxmax, float *tnear, float *tfar)
{
    // compute intersection of ray with all six bbox planes
    vec3 invR = (1.0f) / r.direction;
    vec3 tbot = invR * (boxmin - r.origin);
    vec3 ttop = invR * (boxmax - r.origin);

    // re-order intersections to find smallest and largest on each axis
    vec3 tmin = vMin(ttop, tbot);
    vec3 tmax = vMax(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = maxElementOf(tmin);
    float smallest_tmax = minElementOf(tmax);

    *tnear = largest_tmin;
    *tfar = smallest_tmax;

    if(*tnear < 0)
    {
        *tnear = 0;
    }

    return smallest_tmax > largest_tmin;
}

struct _KernelLaunchInfo
{
    integer3 threadIdx;
    integer3 blockDim;
    integer3 blockIdx;
    // integer3 gridDim;
};

#ifdef __CUDACC__
typedef void *KernelLaunchInfo;
#else
typedef _KernelLaunchInfo KernelLaunchInfo;
#endif

template <class T>
struct Volume
{
    int m_xSize;
    int m_ySize;
    int m_zSize;
    int m_xySize;
    int m_total;
    T *m_data;

    HOST DEVICE Volume(int x, int y, int z)
//         : m_xSize(x), m_ySize(y), m_zSize(z), m_xySize(x * y), m_total(x * y * z)
    {
//         m_data = dataRef;
    }
    HOST DEVICE ~Volume() {}
    HOST DEVICE inline int index(int i, int j, int k) const
    {
        return i + j * m_xSize + k * m_xySize;
    }

    HOST DEVICE T sampleNearestAccurately(vec3 pos) const
        // pos is normalized coordinates in object space, assumed to be in [0,1]^3
    {
        int i = pos.x * m_xSize;
        int j = pos.y * m_ySize;
        int k = pos.z * m_zSize;

        i = iMax(0, iMin(m_xSize - 1, i));
        j = iMax(0, iMin(m_ySize - 1, j));
        k = iMax(0, iMin(m_zSize - 1, k));

        return m_data[index(i, j, k)];
    }
    HOST DEVICE T sampleLinearAccurately(vec3 pos) const
        // pos is normalized coordinates in object space, assumed to be in [0,1]^3
    {
        float i = pos.x * m_xSize - 0.5f;
        float j = pos.y * m_ySize - 0.5f;
        float k = pos.z * m_zSize - 0.5f;

        int i0 = int(i);  // might need strict_floor()
        int j0 = int(j);  // might need strict_floor()
        int k0 = int(k);  // might need strict_floor()
        int i1 = i0 + 1;
        int j1 = j0 + 1;
        int k1 = k0 + 1;

        i0 = iMax(0, iMin(m_xSize - 1, i0));
        j0 = iMax(0, iMin(m_ySize - 1, j0));
        k0 = iMax(0, iMin(m_zSize - 1, k0));
        i1 = iMax(0, iMin(m_xSize - 1, i1));
        j1 = iMax(0, iMin(m_ySize - 1, j1));
        k1 = iMax(0, iMin(m_zSize - 1, k1));

        float s1 = i - i0;
        float t1 = j - j0;
        float u1 = k - k0;
        float s0 = 1 - s1;
        float t0 = 1 - t1;
        float u0 = 1 - u1;

        return 
            s0 * t0 * u0 * m_data[index(i0, j0, k0)] +
            s0 * t0 * u1 * m_data[index(i0, j0, k1)] +
            s0 * t1 * u0 * m_data[index(i0, j1, k0)] +
            s0 * t1 * u1 * m_data[index(i0, j1, k1)] +
            s1 * t0 * u0 * m_data[index(i1, j0, k0)] +
            s1 * t0 * u1 * m_data[index(i1, j0, k1)] +
            s1 * t1 * u0 * m_data[index(i1, j1, k0)] +
            s1 * t1 * u1 * m_data[index(i1, j1, k1)] ;
    }
    HOST DEVICE vec3 getNormalizedObjectCoordinates(int i, int j, int k) const
    {
        return vec3((i + 0.5f) / m_xSize, (j + 0.5f) / m_ySize, (k + 0.5f) / m_zSize);
    }
    HOST DEVICE void setVoxel(int i, int j, int k, const T& val)
    {
        m_data[index(i, j, k)] = val;
    }
    HOST DEVICE T getVoxel(int i, int j, int k) const
    {
        return m_data[index(i, j, k)];
    }

protected:
    HOST DEVICE Volume()
        : m_xSize(0), m_ySize(0), m_zSize(0), m_xySize(0), m_total(0), m_data(NULL)
    {
    }
};

template<class T>
struct VolumeProxy : public Volume<T>
{
    HOST DEVICE VolumeProxy(int x, int y, int z, T *dataRef)
    {
        m_xSize = x;
        m_ySize = y;
        m_zSize = z;
        m_xySize = x * y;
        m_total = x * y * z;
        m_data = dataRef;
    }
    HOST DEVICE ~VolumeProxy() {}
};

#ifdef __CUDACC__
template<class T>
struct VolumeDevice : public Volume<T>
{
    HOST VolumeDevice(int x, int y, int z)
    {
        m_xSize = x;
        m_ySize = y;
        m_zSize = z;
        m_xySize = x * y;
        m_total = x * y * z;

        checkCudaErrors(cudaMalloc((void**)&m_data, sizeof(T) * m_total));
    }
    HOST ~VolumeDevice()
    {
        checkCudaErrors(cudaFree(m_data));
    }
    HOST void copyFrom(T *dataHost)
    {
        checkCudaErrors(cudaMemcpy(m_data, dataHost, sizeof(T) * m_total, cudaMemcpyHostToDevice));
    }
    HOST void copyTo(T *dataHost)
    {
        checkCudaErrors(cudaMemcpy(dataHost, m_data, sizeof(T) * m_total, cudaMemcpyDeviceToHost));
    }

    HOST VolumeProxy<T> getProxy()
    {
        return VolumeProxy<T>(m_xSize, m_ySize, m_zSize, m_data);
    }

protected:
    HOST VolumeDevice()
//         : m_xSize(0), m_ySize(0), m_zSize(0), m_xySize(0), m_total(0), m_data(NULL)
    {}
};
#else
template<class T>
struct VolumeDevice : public Volume<T>
{
    HOST VolumeDevice(int x, int y, int z)
    {
        m_xSize = x;
        m_ySize = y;
        m_zSize = z;
        m_xySize = x * y;
        m_total = x * y * z;

        m_data = new T[m_total];
    }
    HOST ~VolumeDevice()
    {
        if(m_data) delete [] m_data;
    }
    HOST void copyFrom(T *dataHost)
    {
        memcpy(m_data, dataHost, sizeof(T) * m_total);
    }
    HOST void copyTo(T *dataHost)
    {
        memcpy(dataHost, m_data, sizeof(T) * m_total);
    }

    HOST VolumeProxy<T> getProxy()
    {
        return VolumeProxy<T>(m_xSize, m_ySize, m_zSize, m_data);
    }

protected:
    HOST VolumeDevice()
        : m_xSize(0), m_ySize(0), m_zSize(0), m_xySize(0), m_total(0), m_data(NULL)
    {}
};
#endif

template<class T>
struct VolumeHost : public Volume<T>
{
    HOST VolumeHost(int x, int y, int z)
    {
        m_xSize = x;
        m_ySize = y;
        m_zSize = z;
        m_xySize = x * y;
        m_total = x * y * z;
        m_data = new T[m_total];
        clearWithValue(0);
    }
    HOST ~VolumeHost()
    {
        if (m_data) delete [] m_data;
    }
    HOST VolumeHost(const char *filename)
    {
        m_data = NULL;
        loadBinaryFile(filename);
    }
    HOST void clearWithValue(T a)
    {
        for (int n = 0; n < m_total; n++)
        {
            m_data[n] = a;
        }
    }
    HOST void binarize(float a)
    {
        for (int n = 0; n < m_total; n++)
        {
            m_data[n] = m_data[n] < a ? 0.0f : 1.0f;
        }
    }
    HOST int getDimSize(int dimensionIndex)
        // using MATLAB convention
    {
        return dimensionIndex == 1 ? m_xSize
             : dimensionIndex == 2 ? m_ySize 
             : dimensionIndex == 3 ? m_zSize
             : -1;
    }
    HOST void permute(int firstDim, int secondDim, int thirdDim)
    {
        if (firstDim < 1 || firstDim > 3 ||
            secondDim < 1 || secondDim > 3 ||
            thirdDim < 1 || thirdDim > 3 ||
            firstDim == secondDim ||
            firstDim == thirdDim ||
            secondDim == thirdDim)
        {
            printf("volume<> - permute() not callable\n");
            return;
        }

        int nx = getDimSize(firstDim);
        int ny = getDimSize(secondDim);
        int nz = getDimSize(thirdDim);
        int nxy = nx * ny;

        int oldPitch[3] = { 1, m_xSize, m_xySize };
        int firstPitch  = oldPitch[firstDim  - 1];
        int secondPitch = oldPitch[secondDim - 1];
        int thirdPitch  = oldPitch[thirdDim  - 1];

        T *m_data_old = new T[m_total];
        memcpy(m_data_old, m_data, sizeof(T) * m_total);
        for (int k = 0; k < nz; k++)
        {
            for (int j = 0; j < ny; j++)
            {
                for (int i = 0; i < nx; i++)
                {
                    m_data[i + j * nx + k * nxy] = 
                        m_data_old[i * firstPitch + j * secondPitch + k * thirdPitch];
                }
            }
        }
        delete [] m_data_old;

        m_xSize = nx;
        m_ySize = ny;
        m_zSize = nz;
        m_xySize = nx * ny;
    }
   
    HOST void checkMinMax();
    HOST void checkNAN();
    HOST void loadBinaryFile(const char *filename);
    HOST void dumpBinaryFile(const char *filename);
    HOST void addFrame(int size);

    HOST VolumeProxy<T> getProxy()
    {
        return VolumeProxy<T>(m_xSize, m_ySize, m_zSize, m_data);
    }

protected:
    HOST VolumeHost()
//         : m_xSize(0), m_ySize(0), m_zSize(0), m_xySize(0), m_total(0), m_data(NULL)
    {
        printf("volume<> - initialized empty\n");
    }
};

HOST void VolumeHost<float>::checkMinMax()
{
    float findMin =   FLT_MAX;
    float findMax = - FLT_MAX;

    printf("m_total: %d\n", m_total);
    for (int n = 0; n < m_total; n++)
    {
        float data = m_data[n];
        if (findMin > data)
        {
            findMin = data;
        }
        if (findMax < data)
        {
            findMax = data;
        }
    }

    printf("min: %f\nmax: %f\n", findMin, findMax);
}
HOST void VolumeHost<vec3>::checkMinMax()
{
    printf("volume<vec3>::checkMinMax() not yet implemented\n");
}
HOST void VolumeHost<float>::checkNAN()
{
    for (int k = 0; k < m_zSize; k++)
    {
        for (int j = 0; j < m_ySize; j++)
        {
            for (int i = 0; i < m_xSize; i++)
            {
                if (m_data[index(i, j, k)] != m_data[index(i, j, k)])
                {
                    printf("NAN: (%d, %d, %d)\n", i, j, k);
                }
            }
        }
    }
}
HOST void VolumeHost<vec3>::checkNAN()
{
    printf("volume<vec3>::checkNAN() not yet implemented\n");
}
HOST void VolumeHost<float>::loadBinaryFile(const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    int xSize, ySize, zSize;
    fread(&xSize, sizeof(int), 1, fp);
    fread(&ySize, sizeof(int), 1, fp);
    fread(&zSize, sizeof(int), 1, fp);

    if(xSize > 0 && ySize > 0 && zSize > 0)
    {
        if (NULL != m_data)
        {
            printf("volume<float> - clearing data\n");
            delete [] m_data;
        }

        m_xSize = xSize;
        m_ySize = ySize;
        m_zSize = zSize;
        m_xySize = m_xSize * m_ySize;
        m_total = m_xSize * m_ySize * m_zSize;
        m_data = new float[m_total];
        fread(m_data, sizeof(float) * m_total, 1, fp);

        printf("volume<float> - loaded %s\n", filename);
    }

    fclose(fp);
}
HOST void VolumeHost<vec3>::loadBinaryFile(const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    int xSize, ySize, zSize;
    fread(&xSize, sizeof(int), 1, fp);
    fread(&ySize, sizeof(int), 1, fp);
    fread(&zSize, sizeof(int), 1, fp);

    if(xSize > 0 && ySize > 0 && zSize > 0)
    {
        if (NULL != m_data)
        {
            printf("volume<vec3> - clearing data\n");
            delete [] m_data;
        }

        m_xSize = xSize;
        m_ySize = ySize;
        m_zSize = zSize;
        m_xySize = m_xSize * m_ySize;
        m_total = m_xSize * m_ySize * m_zSize;
        m_data = new vec3[m_total];
        fread(m_data, sizeof(vec3) * m_total, 1, fp);

        printf("volume<vec3> - loaded %s\n", filename);
    }

    fclose(fp);
}
HOST void VolumeHost<float>::dumpBinaryFile(const char *filename)
{
    if(m_xSize > 0 && m_ySize > 0 && m_zSize > 0 && NULL != m_data)
    {
        FILE *fp = fopen(filename, "wb");
        fwrite(&m_xSize, sizeof(int), 1, fp);
        fwrite(&m_ySize, sizeof(int), 1, fp);
        fwrite(&m_zSize, sizeof(int), 1, fp);

        fwrite(m_data, sizeof(float) * m_total, 1, fp);

        printf("volume<float> - dumped %s\n", filename);
        fclose(fp);
    }
}
HOST void VolumeHost<vec3>::dumpBinaryFile(const char *filename)
{
    if(m_xSize > 0 && m_ySize > 0 && m_zSize > 0 && NULL != m_data)
    {
        FILE *fp = fopen(filename, "wb");
        fwrite(&m_xSize, sizeof(int), 1, fp);
        fwrite(&m_ySize, sizeof(int), 1, fp);
        fwrite(&m_zSize, sizeof(int), 1, fp);

        fwrite(m_data, sizeof(vec3) * m_total, 1, fp);

        printf("volume<vec3> - dumped %s\n", filename);
        fclose(fp);
    }
}
HOST void VolumeHost<float>::addFrame(int size)
{
    // i == 0 and i == Nx - 1 face
    for (int k = 0; k < m_zSize; k++)
    {
        for (int j = 0; j < m_ySize; j++)
        {
            if (j < size || j >= m_ySize - size ||
                k < size || k >= m_zSize - size)
            {
                m_data[index(          0, j, k)] = 1.0f;
                m_data[index(m_xSize - 1, j, k)] = 1.0f;
            }
        }
    }

    // j
    for (int k = 0; k < m_zSize; k++)
    {
        for (int i = 0; i < m_xSize; i++)
        {
            if (i < size || i >= m_xSize - size ||
                k < size || k >= m_zSize - size)
            {
                m_data[index(i,           0, k)] = 1.0f;
                m_data[index(i, m_ySize - 1, k)] = 1.0f;
            }
        }
    }

    // k
    for (int j = 0; j < m_ySize; j++)
    {
        for (int i = 0; i < m_xSize; i++)
        {
            if (i < size || i >= m_xSize - size ||
                j < size || j >= m_ySize - size)
            {
                m_data[index(i, j,           0)] = 1.0f;
                m_data[index(i, j, m_zSize - 1)] = 1.0f;
            }
        }
    }
}
HOST void VolumeHost<vec3>::addFrame(int size)
{
    printf("volume<vec3>::addFrame() not yet implemented\n");
}

struct camera
{
    vec3 m_posEye;
    vec3 m_dirForward;
    vec3 m_dirRight;
    vec3 m_dirUp;
    float m_tangentHalfFovY;
    
    HOST DEVICE camera()
        : m_posEye(0, 0, 0),
          m_dirForward(0, 0, -1),
          m_dirRight(1, 0, 0),
          m_dirUp(0, 1, 0)
    {
        setFovY(30);
    }
    HOST DEVICE void setPosEye(float x, float y, float z)
    {
        m_posEye.x = x;
        m_posEye.y = y;
        m_posEye.z = z;
    }
    HOST DEVICE void setDirForward(float x, float y, float z)
    {
        m_dirForward.x = x;
        m_dirForward.y = y;
        m_dirForward.z = z;
    }
    HOST DEVICE void setDirRight(float x, float y, float z)
    {
        m_dirRight.x = x;
        m_dirRight.y = y;
        m_dirRight.z = z;
    }
    HOST DEVICE void setDirUp(float x, float y, float z)
    {
        m_dirUp.x = x;
        m_dirUp.y = y;
        m_dirUp.z = z;
    }
    HOST DEVICE void setFovY(float fovy)
    {
        m_tangentHalfFovY = tan(fovy * 0.5f * M_PI / 180.0f);
    }
    HOST DEVICE ray makeRay(float x, float y)
        // expect y to be in [0,1] and x to be in [0, aspect ratio]
    {
        float x1 = (x) * 2 - 1;
        float y1 = (y) * 2 - 1;

        vec3 x_shift = m_dirRight * x1 * m_tangentHalfFovY;
        vec3 y_shift = m_dirUp    * y1 * m_tangentHalfFovY;

        return ray(m_posEye, normalize(m_dirForward + x_shift + y_shift));
    }
};

struct Film
{
    int m_filmWidth;
    int m_filmHeight;
    int m_total;

    vec3 *m_buffer;

    HOST DEVICE Film(int w, int h)
        : m_filmWidth(w), m_filmHeight(h), m_total(0), m_buffer(NULL)
    {
    }
    HOST DEVICE ~Film()
    {
    }
    HOST DEVICE void setPixel(int i, int j, vec3 rgb)
    {
        m_buffer[i + j * m_filmWidth] = rgb;
    }
    HOST DEVICE vec3 getPixel(int i, int j)
    {
        return m_buffer[i + j * m_filmWidth];
    }
protected:
    HOST DEVICE Film() {}
};

struct FilmProxy : public Film
{
    HOST DEVICE FilmProxy(int w, int h, vec3 *bufferRef)
    {
        m_filmWidth = w;
        m_filmHeight = h;
        m_total = m_filmWidth * m_filmHeight;

        m_buffer = bufferRef;
    }
    HOST DEVICE ~FilmProxy() {}
};

#ifdef __CUDACC__
struct FilmDevice : public Film
{
    HOST FilmDevice(int w, int h)
    {
        m_filmWidth = w;
        m_filmHeight = h;
        m_total = m_filmWidth * m_filmHeight;

        checkCudaErrors(cudaMalloc((void**)&m_buffer, sizeof(vec3) * m_total));
    }
    HOST ~FilmDevice()
    {
        checkCudaErrors(cudaFree(m_buffer));
    }
    HOST void copyFrom(vec3 *dataHost)
    {
        checkCudaErrors(cudaMemcpy(m_buffer, dataHost, sizeof(vec3) * m_total, cudaMemcpyHostToDevice));
    }
    HOST void copyTo(vec3 *dataHost)
    {
        checkCudaErrors(cudaMemcpy(dataHost, m_buffer, sizeof(vec3) * m_total, cudaMemcpyDeviceToHost));
    }

    HOST FilmProxy getProxy()
    {
        return FilmProxy(m_filmWidth, m_filmHeight, m_buffer);
    }

protected:
    HOST FilmDevice() {}
};
#else
struct FilmDevice : public Film
{
    HOST FilmDevice(int w, int h)
    {
        m_filmWidth = w;
        m_filmHeight = h;
        m_total = m_filmWidth * m_filmHeight;

        m_buffer = new vec3[m_total];
    }
    HOST ~FilmDevice()
    {
        if (m_buffer) delete [] m_buffer;
    }
    HOST void copyFrom(vec3 *dataHost)
    {
        memcpy(m_buffer, dataHost, sizeof(vec3) * m_total);
    }
    HOST void copyTo(vec3 *dataHost)
    {
        memcpy(dataHost, m_buffer, sizeof(vec3) * m_total);
    }

    HOST FilmProxy getProxy()
    {
        return FilmProxy(m_filmWidth, m_filmHeight, m_buffer);
    }

protected:
    HOST FilmDevice() {}
};
#endif

struct FilmHost : public Film
{
    HOST FilmHost(int w, int h)
    {
        m_filmWidth = w;
        m_filmHeight = h;
        m_total = m_filmWidth * m_filmHeight;
        m_buffer = new vec3[m_total];
        resetBlack();
    }
    HOST ~FilmHost()
    {
        if (m_buffer) delete [] m_buffer;
    }
    HOST void resetBlack()
    {
        for (int n = 0; n < m_total; ++n)
        {
            m_buffer[n] = vec3();
        }
    }
    HOST void resetRandom()
    {
        for (int n = 0; n < m_total; ++n)
        {
            m_buffer[n] = vec3(randf(), randf(), randf());
        }
    }
    HOST void resetRandomGray()
    {
        for (int n = 0; n < m_total; ++n)
        {
            m_buffer[n] = vec3(randf());
        }
    }
    HOST void outputPPM(const char *filename)
    {
        FILE *fp = fopen(filename, "wb");
        fprintf(fp, "P3\n%d %d\n255\n",m_filmWidth, m_filmHeight);
        for (int j = m_filmHeight - 1; j >= 0; --j)
        {
            for (int i = 0; i < m_filmWidth; i++)
            {
                int n = i + j * m_filmWidth;
                fprintf(fp, "%d %d %d\n", toInt(m_buffer[n].x), toInt(m_buffer[n].y), toInt(m_buffer[n].z));
            }
        }
        fclose(fp);
    }
    HOST void outputBIN(const char *filename)
    {
        FILE *fp = fopen(filename, "wb");
        fwrite(&m_filmWidth, sizeof(int), 1, fp);
        fwrite(&m_filmHeight, sizeof(int), 1, fp);

        vec3 *tempBuffer = new vec3[m_total];
        for (int j = 0; j < m_filmHeight; j++)
        {
            for (int i = 0; i < m_filmWidth; i++)
            {
                tempBuffer[i + j * m_filmWidth] = m_buffer[i + (m_filmHeight - 1 - j) * m_filmWidth];
            }
        }
        fwrite((void *)tempBuffer, sizeof(vec3), m_total, fp);
        delete [] tempBuffer;
        fclose(fp);

    }

    HOST FilmProxy getProxy()
    {
        return FilmProxy(m_filmWidth, m_filmHeight, m_buffer);
    }

protected:
    HOST FilmHost() {}
};

class VolumeProperties
{
    float m_sigmaEpsilon;
    vec3 sigmaS;
    vec3 sigmaA;
    vec3 sigmaT;
    vec3 albedo;
public:
    HOST DEVICE VolumeProperties(float epsilon)
    {
        m_sigmaEpsilon = epsilon;

        sigmaS = MEDIUM_SCALE * vec3(0.70f, 1.22f, 1.90f);
        sigmaA = MEDIUM_SCALE * vec3(0.0014f, 0.0025f, 0.0142f);
        sigmaT = sigmaS + sigmaA;
        albedo = sigmaS / sigmaT;
    }
    HOST DEVICE inline vec3 getSigmaT(float density)
    {
        return vMax(m_sigmaEpsilon, density * sigmaT);
    }
    HOST DEVICE inline vec3 getSigmaS(float density)
    {
        return density * sigmaS;
    }
    HOST DEVICE inline vec3 getSigmaA(float density)
    {
        return density * sigmaA;
    }
    HOST DEVICE inline vec3 getAlbedo()
    {
        return albedo;
    }
};

float lightPower = M_FOUR_PI;

struct volumeContainer
{
    vec3 m_worldSpaceMin;
    vec3 m_worldSpaceMax;
    vec3 m_inverseScale;

    HOST DEVICE volumeContainer(vec3 wMin, vec3 wMax)
    {
        m_worldSpaceMin = wMin;
        m_worldSpaceMax = wMax;

        if (minElementOf(m_worldSpaceMax - m_worldSpaceMin) == 0.0f)
        {
            // printf("boundingBox - singularity detected\n");
            m_worldSpaceMax = m_worldSpaceMin + vec3(1e-6f);
        }
        m_inverseScale = 1.0f / (m_worldSpaceMax - m_worldSpaceMin);
    }

    HOST DEVICE vec3 transformToObjectSpace(vec3 worldCoord) const
    {
        return vec3((worldCoord - m_worldSpaceMin) * m_inverseScale);
    }
    HOST DEVICE vec3 transformToWorldSpace(vec3 objectCoord) const
    {
        return m_worldSpaceMin + objectCoord * (m_worldSpaceMax - m_worldSpaceMin);
    }
    HOST DEVICE vec3 getExtent() const
    {
        return m_worldSpaceMax - m_worldSpaceMin;
    }
};

struct Light
{
    vec3 directionalLight;
    float directionalLightPower;

    Light()
    {
        directionalLight = normalize(vec3(1, 0, 0));
        directionalLightPower = 10.0f; // white light
    }
};

DEVICE vec3 raymarch(ray eyeRay, volumeContainer boundingBox, VolumeProxy<float> volDensity, VolumeProxy<vec3> reducedLight,
        VolumeProxy<vec3> volFluence, VolumeProperties& params)
{
    float dt = 0.1f * STEP_SIZE_REFINE;

    vec3 throughput(1);
    vec3 radiance(0);

    float tNear, tFar;

    bool hitBox = CUDASDK_intersectBox(eyeRay, boundingBox.m_worldSpaceMin, boundingBox.m_worldSpaceMax, &tNear, &tFar);

    if (hitBox)
    {
        for (float t = tNear; t < tFar; t += dt)
        {
            vec3 objectSpaceSamplingPosition(boundingBox.transformToObjectSpace(eyeRay.proceed(t)));
            float density = volDensity.LINEAR_INTERPOLATOR(objectSpaceSamplingPosition);
            throughput *= exp( - params.getSigmaT(density) * dt);

            // Q(x) = 1/4pi * (q_ri(x) + sigma_s(x) * phi(x) + j(x), see equation (8)
            // but here only includes the external ones: Q_e(x) = 1/4pi * (q_ri(x) + j(x))
            float phaseFunction = M_ONE_OVER_FOUR_PI;

            // use the density field (original resolution)
            vec3 localSigmaS = params.getSigmaS(density);

            vec3 localSource = phaseFunction * ( reducedLight.LINEAR_INTERPOLATOR(objectSpaceSamplingPosition)
                +  localSigmaS * volFluence.LINEAR_INTERPOLATOR(objectSpaceSamplingPosition) );

            radiance += throughput * localSource * dt;
        }
    }
    return radiance;
}

KERNEL void _renderByRaymarching(camera cam, FilmProxy fm, volumeContainer boundingBox, VolumeProxy<float> volDensity, VolumeProxy<vec3> reducedLight,
    VolumeProxy<vec3> volFluence, VolumeProperties params, KernelLaunchInfo info)
{
#ifdef __CUDACC__
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
#else
    int i = info.threadIdx.x + info.blockIdx.x * info.blockDim.x;
    int j = info.threadIdx.y + info.blockIdx.y * info.blockDim.y;
#endif

    if (i >= fm.m_filmWidth || j >= fm.m_filmHeight) return;

    ray primary_ray = cam.makeRay( (i + 0.5f) / fm.m_filmWidth, (j + 0.5f) / fm.m_filmHeight);

    fm.setPixel(i, j, raymarch(primary_ray, boundingBox, volDensity, reducedLight, volFluence, params));
}

void renderByRaymarching(camera &cam, FilmHost &fm, volumeContainer *boundingBox, 
    VolumeDevice<float> *inputPyroCloud, 
    VolumeDevice<vec3> *reducedIncidentLight,
    VolumeDevice<vec3> *volFluence, 
    VolumeProperties *params, const char *filename)
{
    const volumeContainer& _boundingBox = *boundingBox;

    FilmDevice d_fm(fm.m_filmWidth, fm.m_filmHeight);

#ifdef __CUDACC__
    dim3 blockConfig(16, 16);
    dim3 gridConfig(divideUp(fm.m_filmWidth, blockConfig.x), divideUp(fm.m_filmHeight, blockConfig.y));
    KernelLaunchInfo info;

    _renderByRaymarching<<<gridConfig, blockConfig>>>(cam, d_fm.getProxy(), _boundingBox, inputPyroCloud->getProxy(), reducedIncidentLight->getProxy(),
        volFluence->getProxy(), *params, info);
#else
    integer3 blockConfig = { 16, 16, 1 };
    integer3 gridConfig = { divideUp(fm.m_filmWidth, blockConfig.x), divideUp(fm.m_filmHeight, blockConfig.y), 1 };

    for (int j = 0; j < fm.m_filmHeight; j++)
    {
#pragma omp parallel for
        for (int i = 0; i < fm.m_filmWidth; i++)
        {
            KernelLaunchInfo info;
            info.threadIdx.x = i % blockConfig.x;
            info.threadIdx.y = j % blockConfig.y;
            info.threadIdx.z = 1;
            info.blockDim = blockConfig;
            info.blockIdx.x = i / blockConfig.x;
            info.blockIdx.y = j / blockConfig.y;
            info.blockIdx.z = 1;

            _renderByRaymarching(cam, d_fm.getProxy(), _boundingBox, inputPyroCloud->getProxy(), reducedIncidentLight->getProxy(),
                volFluence->getProxy(), *params, info);
        }
        printf("\rRaymarching %.2f %%", (j) / float(fm.m_filmHeight - 1) * 100.0f);
    }
#endif

    d_fm.copyTo(fm.m_buffer);
    fm.outputPPM(filename);
}

HOST DEVICE vec3 fluxLimiter(vec3 knudsenNumber)
    // note the possible singularity
{
    //LP
    {
        // causes numerical singularity
        //     vec3 oneOverR = 1.0f / knudsenNumber;
        //     vec3 exponential = exp(2.0f * knudsenNumber);
        //     vec3 cothR = (exponential + 1.0f) / (exponential - 1.0f);
        //     return oneOverR * (cothR - oneOverR);

        // this form of coth works fine
        // knudsenNumber = vMax(knudsenNumber, 0.01f); // to enforce accuracy for single precision
        // vec3 oneOverR = 1.0f / knudsenNumber;
        // vec3 exponential = exp( - 2.0f * knudsenNumber);
        // vec3 cothR = (1 + exponential) / (1 - exponential);
        // return oneOverR * (cothR - oneOverR);
    }

//     return 1.0f / 3.0f; // CDA                                                          //onethird
//     return 1.0f / (3.0f + knudsenNumber);                                            //sum
    return 1.0f / vMax(3.0f, knudsenNumber);//+++++++++++++++++++++++++++++++++      //max
//     return 2.0f / (3.0f + sqrt(9.0f + 4.0f * knudsenNumber * knudsenNumber));        //kershaw
//     return pow(powf(3.0, 2.0) + pow(knudsenNumber, 2.0), -1/2.0);                    //larsen
}

#define FOR_ALL_INNER_CELLS_BEGIN   for (int k = 1; k < nz - 1; k++)          \
                                    {                                         \
                                        for (int j = 1; j < ny - 1; j++)      \
                                        {                                     \
                                            for (int i = 1; i < nx - 1; i++)  \
                                            {
#define FOR_ALL_INNER_CELLS_END }}}


#define FOR_ALL_CELLS_BEGIN   for (int k = 0; k < nz; k++)                    \
                              {                                               \
                                  for (int j = 0; j < ny; j++)                \
                                  {                                           \
                                      for (int i = 0; i < nx; i++)            \
                                      {
#define FOR_ALL_CELLS_END }}}


KERNEL void _computeDiffusionCoefficients(
    VolumeProxy<float> volDensityResampled,
    VolumeProxy<vec3> volFluence, 
    VolumeProxy<vec3> volDiffusionCoefficients,
    VolumeProperties params, 
    float oneOverTwoDeltaL, KernelLaunchInfo info)
{
#ifdef __CUDACC__
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
#else
    int i = info.threadIdx.x + info.blockIdx.x * info.blockDim.x;
    int j = info.threadIdx.y + info.blockIdx.y * info.blockDim.y;
    int k = info.threadIdx.z + info.blockIdx.z * info.blockDim.z;
#endif

    // do not update the boundary
    int nx = volFluence.m_xSize;
    int ny = volFluence.m_ySize;
    int nz = volFluence.m_zSize;
    if (i < 1 || i >= nx - 1 || 
        j < 1 || j >= ny - 1 ||
        k < 1 || k >= nz - 1) return;

    float epsilon = 1e-20f; // fixed and TODO: refer to the paper

    float density = volDensityResampled.getVoxel(i, j, k);
    vec3 gradPhiX = (volFluence.getVoxel(i + 1, j, k) - volFluence.getVoxel(i - 1, j, k)) * oneOverTwoDeltaL;
    vec3 gradPhiY = (volFluence.getVoxel(i, j + 1, k) - volFluence.getVoxel(i, j - 1, k)) * oneOverTwoDeltaL;
    vec3 gradPhiZ = (volFluence.getVoxel(i, j, k + 1) - volFluence.getVoxel(i, j, k - 1)) * oneOverTwoDeltaL;
    vec3 knudsenNumber = 
        vMax(sqrt(gradPhiX * gradPhiX + gradPhiY * gradPhiY + gradPhiZ * gradPhiZ), epsilon) /
        vMax(params.getSigmaT(density) * volFluence.getVoxel(i, j, k), epsilon);

    // debugPrint(fluxLimiter(knudsenNumber));

    vec3 diffusionCoefficient = fluxLimiter(knudsenNumber) / params.getSigmaT(density);
    volDiffusionCoefficients.setVoxel(i, j, k, diffusionCoefficient);
}

KERNEL void _solveDiffusionEquation(
    VolumeProxy<float> volDensityResampled,
    VolumeProxy<vec3> volSource, 
    VolumeProxy<vec3> volFluence, 
    VolumeProxy<vec3> volDiffusionCoefficients,
    VolumeProperties params, float deltaL, 
    int turn, KernelLaunchInfo info)
{
#ifdef __CUDACC__
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
#else
    int i = info.threadIdx.x + info.blockIdx.x * info.blockDim.x;
    int j = info.threadIdx.y + info.blockIdx.y * info.blockDim.y;
    int k = info.threadIdx.z + info.blockIdx.z * info.blockDim.z;
#endif

    // do not update the boundary
    int nx = volFluence.m_xSize;
    int ny = volFluence.m_ySize;
    int nz = volFluence.m_zSize;
    if (i < 1 || i >= nx - 1 || 
        j < 1 || j >= ny - 1 ||
        k < 1 || k >= nz - 1) return;

    float omega = 1.0f; // SOR factor

    if((i + j + k) % 2 == turn)
    {
        vec3 diffusionCoefficient = volDiffusionCoefficients.getVoxel(i, j, k);
#if HELPFUL_BUG
        diffusionCoefficient = 1.0 / (3.0 * params.getSigmaT(volDensityResampled.getVoxel(i, j, k))); // gives rise to some interesting artifacts
#endif

        // {
            // vec3 no_limit_Coefficient = 1.0 / (3.0 * params.getSigmaT(volDensityResampled.getVoxel(i, j, k))); // gives rise to some interesting artifacts
            // float a = 1e-8f;
            // diffusionCoefficient = diffusionCoefficient * (1-a) + no_limit_Coefficient * a;
        // }


        vec3 D_ps[6] = 
        {
            (diffusionCoefficient + volDiffusionCoefficients.getVoxel(i + 1, j, k)) * 0.5f,
            (diffusionCoefficient + volDiffusionCoefficients.getVoxel(i - 1, j, k)) * 0.5f,
            (diffusionCoefficient + volDiffusionCoefficients.getVoxel(i, j + 1, k)) * 0.5f,
            (diffusionCoefficient + volDiffusionCoefficients.getVoxel(i, j - 1, k)) * 0.5f,
            (diffusionCoefficient + volDiffusionCoefficients.getVoxel(i, j, k + 1)) * 0.5f,
            (diffusionCoefficient + volDiffusionCoefficients.getVoxel(i, j, k - 1)) * 0.5f,
        };

        float density = volDensityResampled.getVoxel(i, j, k);
        vec3 updatedFluence = 
            ( volSource.getVoxel(i, j, k) * (deltaL * deltaL)
            + D_ps[0] * volFluence.getVoxel(i + 1, j, k)
            + D_ps[1] * volFluence.getVoxel(i - 1, j, k)
            + D_ps[2] * volFluence.getVoxel(i, j + 1, k)
            + D_ps[3] * volFluence.getVoxel(i, j - 1, k)
            + D_ps[4] * volFluence.getVoxel(i, j, k + 1)
            + D_ps[5] * volFluence.getVoxel(i, j, k - 1) ) /
            ( params.getSigmaA(density) * (deltaL * deltaL)
            + D_ps[0] + D_ps[1] + D_ps[2] + D_ps[3] + D_ps[4] + D_ps[5] );

        vec3 oldFluence = volFluence.getVoxel(i, j, k);
        updatedFluence = omega * updatedFluence + (1.0f - omega) * oldFluence;

        if (updatedFluence.x != updatedFluence.x)   updatedFluence.x = 0;
        if (updatedFluence.y != updatedFluence.y)   updatedFluence.y = 0;
        if (updatedFluence.z != updatedFluence.z)   updatedFluence.z = 0;

        volFluence.setVoxel(i, j, k, updatedFluence);
    }
}

void solveDiffusionEquation(volumeContainer *volBoundingBox, 
    VolumeDevice<float> *volDensityResampled,
    VolumeDevice<vec3> *volSource,
    VolumeDevice<vec3> *volFluence, 
    VolumeDevice<vec3> *volDiffusionCoefficients,
    VolumeProperties *params)
    // all volumes are assumed to be of the same size
{
    int nx = volFluence->m_xSize;
    int ny = volFluence->m_ySize;
    int nz = volFluence->m_zSize;

    float deltaL = volBoundingBox->getExtent().x / nx;
    float oneOverTwoDeltaL = 1.0f / (2.0f * deltaL);

#ifdef __CUDACC__
    dim3 blockConfig(8, 8, 8);
    dim3 gridConfig(divideUp(nx, blockConfig.x), divideUp(ny, blockConfig.y), divideUp(nz, blockConfig.z));
    KernelLaunchInfo info;

    _computeDiffusionCoefficients<<<gridConfig, blockConfig>>>(volDensityResampled->getProxy(),
        volFluence->getProxy(), volDiffusionCoefficients->getProxy(), *params, oneOverTwoDeltaL, info);

    for (int turn = 0; turn < 2; turn++)
    {
        _solveDiffusionEquation<<<gridConfig, blockConfig>>>(volDensityResampled->getProxy(), volSource->getProxy(),
            volFluence->getProxy(), volDiffusionCoefficients->getProxy(), *params, deltaL, turn, info);
    }
#else
    integer3 blockConfig = { 8, 8, 8 };
    integer3 gridConfig = { divideUp(nx, blockConfig.x), divideUp(ny, blockConfig.y), divideUp(nz, blockConfig.z) };

    {
#pragma omp parallel for
        FOR_ALL_CELLS_BEGIN
            KernelLaunchInfo info;
            info.threadIdx.x = i % blockConfig.x;
            info.threadIdx.y = j % blockConfig.y;
            info.threadIdx.z = k % blockConfig.z;
            info.blockDim = blockConfig;
            info.blockIdx.x = i / blockConfig.x;
            info.blockIdx.y = j / blockConfig.y;
            info.blockIdx.z = k / blockConfig.z;

            _computeDiffusionCoefficients(volDensityResampled->getProxy(),
                volFluence->getProxy(), volDiffusionCoefficients->getProxy(), *params, oneOverTwoDeltaL, info);
        FOR_ALL_CELLS_END
    }

    for (int turn = 0; turn < 2; turn++)
    {
#pragma omp parallel for
        FOR_ALL_CELLS_BEGIN
            KernelLaunchInfo info;
            info.threadIdx.x = i % blockConfig.x;
            info.threadIdx.y = j % blockConfig.y;
            info.threadIdx.z = k % blockConfig.z;
            info.blockDim = blockConfig;
            info.blockIdx.x = i / blockConfig.x;
            info.blockIdx.y = j / blockConfig.y;
            info.blockIdx.z = k / blockConfig.z;

            _solveDiffusionEquation(volDensityResampled->getProxy(), volSource->getProxy(),
                volFluence->getProxy(), volDiffusionCoefficients->getProxy(), *params, deltaL, turn, info);
        FOR_ALL_CELLS_END
    }
#endif
}

KERNEL void _preComputeSource(Light light,
    volumeContainer volBoundingBox,
    VolumeProxy<float> volDensity, 
    VolumeProxy<vec3> volReducedLight,
    VolumeProperties params,
    KernelLaunchInfo info)
{
#ifdef __CUDACC__
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
#else
    int i = info.threadIdx.x + info.blockIdx.x * info.blockDim.x;
    int j = info.threadIdx.y + info.blockIdx.y * info.blockDim.y;
    int k = info.threadIdx.z + info.blockIdx.z * info.blockDim.z;
#endif

    // do not update the boundary
    int nx = volDensity.m_xSize;
    int ny = volDensity.m_ySize;
    int nz = volDensity.m_zSize;
    if (i < 0 || i >= nx || 
        j < 0 || j >= ny ||
        k < 0 || k >= nz) return;

    float dt = 0.1f * STEP_SIZE_REFINE;

    vec3 start = volReducedLight.getNormalizedObjectCoordinates(i, j, k);
    start = volBoundingBox.transformToWorldSpace(start);
    ray shadowRay(start, light.directionalLight);
    float tNear, tFar;
    bool hit = CUDASDK_intersectBox(shadowRay, volBoundingBox.m_worldSpaceMin, volBoundingBox.m_worldSpaceMax, &tNear, &tFar);

    if (hit)
    {
        vec3 opticalDepth(0);
        for (float t = tNear; t < tFar; t += dt)
        {
            vec3 sampleLocationInWorld(shadowRay.proceed(t));

            // both working, but which one is more accurate ?? because the step size is the same
            vec3 localSigmaT = params.getSigmaT(volDensity.LINEAR_INTERPOLATOR(
                volBoundingBox.transformToObjectSpace(sampleLocationInWorld)));

            opticalDepth += localSigmaT;
        }
        opticalDepth *= dt;
        vec3 throughput = exp( - opticalDepth);

        // normally tNear is 0, so: start == shadowRay.proceed(tNear)
        // also note that sigma_s = albedo * sigma_t
        vec3 localSigmaS = params.getSigmaS(volDensity.getVoxel(i, j, k));

        volReducedLight.setVoxel(i, j, k, light.directionalLightPower * localSigmaS * throughput);
    }
}

void preComputeSource(Light light,
    volumeContainer *volBoundingBox, 
    VolumeDevice<float> *volDensity, 
    VolumeDevice<vec3> *volReducedLight,
    VolumeProperties *params)
{
    int nx = volReducedLight->m_xSize;
    int ny = volReducedLight->m_ySize;
    int nz = volReducedLight->m_zSize;

#ifdef __CUDACC__
    dim3 blockConfig(8, 8, 8);
    dim3 gridConfig(divideUp(nx, blockConfig.x), divideUp(ny, blockConfig.y), divideUp(nz, blockConfig.z));
    KernelLaunchInfo info;

    printf("\rBaking\n");
    _preComputeSource<<<gridConfig, blockConfig>>>(light, *volBoundingBox, volDensity->getProxy(),
        volReducedLight->getProxy(), *params, info);
    checkCudaErrors(cudaDeviceSynchronize());
#else
    integer3 blockConfig = { 8, 8, 8 };
    integer3 gridConfig = { divideUp(nx, blockConfig.x), divideUp(ny, blockConfig.y), divideUp(nz, blockConfig.z) };

    for (int k = 0; k < nz; k++)
    {
        printf("\rBaking %f %%", float(k) / (nx - 1.0f) * 100.0f);
#pragma omp parallel for
        for (int j = 0; j < ny; j++)
        {
            for (int i = 0; i < nx; i++)
            {
                KernelLaunchInfo info;
                info.threadIdx.x = i % blockConfig.x;
                info.threadIdx.y = j % blockConfig.y;
                info.threadIdx.z = k % blockConfig.z;
                info.blockDim = blockConfig;
                info.blockIdx.x = i / blockConfig.x;
                info.blockIdx.y = j / blockConfig.y;
                info.blockIdx.z = k / blockConfig.z;

                _preComputeSource(light, *volBoundingBox, volDensity->getProxy(),
                    volReducedLight->getProxy(), *params, info);
            }
        }
    }
    printf("\n");
#endif
}

template<typename T>
void resampleVolume(VolumeHost<T> *volDensity, VolumeHost<T> *volDensityResampled)
    // assume that volSigmaT and volAlbedo share the same size, but not necessarily with volDensity 
{
    int nx = volDensityResampled->m_xSize;
    int ny = volDensityResampled->m_ySize;
    int nz = volDensityResampled->m_zSize;

    FOR_ALL_CELLS_BEGIN
        vec3 objectSpaceCoordinates = volDensityResampled->getNormalizedObjectCoordinates(i, j, k);
        T density = volDensity->LINEAR_INTERPOLATOR(objectSpaceCoordinates);
        volDensityResampled->setVoxel(i, j, k, density);
    FOR_ALL_CELLS_END
}

int main()
{
//     vec3 boxShift(0, 0, -5);
//     vec3 boundingBoxMin = vec3(-1, -1, -1) + boxShift;
//     vec3 boundingBoxMax = vec3( 1,  1,  1) + boxShift;
    float L = 100.0f;
    vec3 boxShift(0, 0, -L);
    vec3 boundingBoxMin = vec3(-L/5, -L/5, -L/5) + boxShift;
    vec3 boundingBoxMax = vec3( L/5,  L/5,  L/5) + boxShift;
    volumeContainer boundingBox(boundingBoxMin, boundingBoxMax);

#ifdef __CUDACC__
    int sizeN = 256;
#else
    int sizeN = 64;
#endif

    VolumeHost<float> inputPyroCloud(sizeN, sizeN, sizeN);
    {
        //     volume<float> inputPyroCloud("pyro256.bin");
//         volume<float> _inputPyroCloud("vol_1.bin");//+++++++++++++++++++++++++
        VolumeHost<float> _inputPyroCloud("volsmoke.bin");//+++++++++++++++++++++++++
//         VolumeHost<float> _inputPyroCloud("bunny_n256.bin");//+++++++++++++++++++++++++
        //     volume<float> inputPyroCloud("300c.bin");
        //     volume<float> inputPyroCloud("bunny_n200sharp.bin");
        //     volume<float> _inputPyroCloud("bunny_n256.bin"); // why baking so slow??
        //     volume<float> inputPyroCloud("dragon_n200sharp.bin");
        //     inputPyroCloud.addFrame(5);
        //     inputPyroCloud.multiplyBy(20.0f);
//         _inputPyroCloud.binarize(0.02f);

        // Resample (in particular, upsample) the input volume to a specified resolution
        resampleVolume(&_inputPyroCloud, &inputPyroCloud);
    }

    // inputPyroCloud.permute(3, 1, 2); // for stanford bunny

    float reduceFactor = 4; 1; 2; //1; ////
    int nx = int(inputPyroCloud.m_xSize / reduceFactor);
    int ny = int(inputPyroCloud.m_ySize / reduceFactor);
    int nz = int(inputPyroCloud.m_zSize / reduceFactor);

    printf("input volume size: %d x %d x %d\n", inputPyroCloud.m_xSize, inputPyroCloud.m_ySize, inputPyroCloud.m_zSize);
    printf("baking volume size: %d x %d x %d\n", nx, ny, nz);

    // the external light sources combined
    // q_ri + j, the combined source { q_ri = 4pi * sigma_s * L_ri <or> q_ri = sigma_s * L_ri } ?? 
    // or is it true for [ directional light ] that L_ri = L_l * exp(-tau) / 4pi, so: q_ri = 4pi * sigma_s * L_ri = L_l * sigma_s * exp(-tau) ??
    // where L_l is the integral power of the light source and there's no need for dividing by R^2 ??
    VolumeHost<vec3> reducedIncidentLightHighRes(
        inputPyroCloud.m_xSize, 
        inputPyroCloud.m_ySize,
        inputPyroCloud.m_zSize);

    VolumeDevice<float> d_inputPyroCloud(inputPyroCloud.m_xSize,
        inputPyroCloud.m_ySize, inputPyroCloud.m_zSize);
    VolumeDevice<vec3> d_reducedIncidentLightHighRes(
        reducedIncidentLightHighRes.m_xSize, 
        reducedIncidentLightHighRes.m_ySize,
        reducedIncidentLightHighRes.m_zSize);

    VolumeProperties params(1e-3f / minElementOf(boundingBox.getExtent())); // the eps is very important for SOR coefficient
    // VolumeProperties params(1e-10f / minElementOf(boundingBox.getExtent())); // the eps is very important for SOR coefficient

    // baking light volume using high-resolution volumes
    {
        d_inputPyroCloud.copyFrom(inputPyroCloud.m_data);

        preComputeSource(Light(), &boundingBox,
            &d_inputPyroCloud, &d_reducedIncidentLightHighRes, &params);

        d_reducedIncidentLightHighRes.copyTo(reducedIncidentLightHighRes.m_data);
    }

    VolumeDevice<vec3> d_volFluence(nx, ny, nz);
    VolumeDevice<vec3> d_volDiffusionCoefficients(nx, ny, nz);

    // FLD solver
    if (1)
    {
        VolumeHost<float> densityResampled(nx, ny, nz);
        resampleVolume(&inputPyroCloud, &densityResampled);

        VolumeDevice<float> d_densityResampled(
            densityResampled.m_xSize,
            densityResampled.m_ySize,
            densityResampled.m_zSize);
        d_densityResampled.copyFrom(densityResampled.m_data);
        
        VolumeHost<vec3> reducedIncidentLight(nx, ny, nz);
        resampleVolume(&reducedIncidentLightHighRes, &reducedIncidentLight);

        VolumeDevice<vec3> d_reducedIncidentLight(
            reducedIncidentLight.m_xSize,
            reducedIncidentLight.m_ySize,
            reducedIncidentLight.m_zSize);
        d_reducedIncidentLight.copyFrom(reducedIncidentLight.m_data);

#ifdef __CUDACC__
        for (int it = 1; it <= 1000 * 10; it++)
#else
        for (int it = 1; it <= 1000; it++)
#endif
        {
            printf("\rSolving %d-th iteration", it);
            solveDiffusionEquation(&boundingBox, &d_densityResampled, 
                &d_reducedIncidentLight, &d_volFluence, &d_volDiffusionCoefficients, &params);

            // monitor the diffusion process
//             if ( (it - 1) % 100 == 0 )
//             {
//                 FilmHost fm(512, 512);
//                 camera cam;
// 
//                 char ppmFilename[256];
//                 sprintf(ppmFilename, "ppm/test_%05d.ppm", it);
//                 renderByRaymarching(cam, fm, &boundingBox, &d_inputPyroCloud, &d_reducedIncidentLightHighRes, // or use the low res equivalent to compare
//                     &d_volFluence, &params, ppmFilename);
//             }
        }
        printf("\n");
    }

//     reducedIncidentLightHighRes.clearWithValue(0); // to check the result with only the contribution from fluence (indirect lighting)
//     volFluence.clearWithValue(0);

#ifdef __CUDACC__
    FilmHost fm(2048, 2048);
#else
    FilmHost fm(512, 512);
#endif
    camera cam;

    const char *ppmFilename = "_test_.ppm";
    renderByRaymarching(cam, fm, &boundingBox, &d_inputPyroCloud, &d_reducedIncidentLightHighRes, // or use the low res equivalent to compare
        &d_volFluence, &params, ppmFilename);

//     fm.outputBIN("_test_.bin");

    return 0;
}

