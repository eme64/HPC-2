#pragma once

#include <cassert>
#include <type_traits>
#include <utility>
#include <algorithm>
#include <typeinfo>
#include <cassert>


/**
 * This container keeps data on the device (GPU) and on the host (CPU)
 *
 * Allocates pinned memory on host, to speed up host-device data migration
 *
 * \rst
 * .. note::
 *    Host and device data are not automatically synchronized!
 *    Use downloadFromDevice() and uploadToDevice() MANUALLY to sync
 * \endrst
 *
 * Never releases any memory, keeps a buffer big enough to
 * store maximum number of elements it ever held
 */
template<typename T>
class PinnedBuffer
{
private:
	int capacity;   ///< Storage buffers size
	int _size;      ///< Number of elements stored now
	T * hostptr;    ///< Host pointer to data
	T * devptr;     ///< Device pointer to data

	/**
	 * Set #_size = \p n. If n > #capacity, allocate more memory
	 * and copy the old data on CUDA stream \p stream (only if \p copy is true)
	 * Copy both host and device data if \p copy is true
	 *
	 * If debug level is high enough, will report cases when the buffer had to grow
	 *
	 * @param n new size, must be >= 0
	 * @param stream data will be copied on that CUDA stream
	 * @param copy if we need to copy old data
	 */
	void _resize(const int n, cudaStream_t stream, bool copy)
	{
		T * hold = hostptr;
		T * dold = devptr;
		int oldsize = _size;

		assert(n >= 0);
		_size = n;
		if (capacity >= n) return;

		const int conservative_estimate = (int)ceil(1.1 * n + 10);
		capacity = 128 * ((conservative_estimate + 127) / 128);

		cudaHostAlloc(&hostptr, sizeof(T) * capacity, 0);
		cudaMalloc(&devptr, sizeof(T) * capacity);

		if (copy && hold != nullptr && oldsize > 0)
		{
			memcpy(hostptr, hold, sizeof(T) * oldsize);
			cudaMemcpyAsync(devptr, dold, sizeof(T) * oldsize, cudaMemcpyDeviceToDevice, stream);
			cudaStreamSynchronize(stream);
		}

		cudaFreeHost(hold);
		cudaFree(dold);
	}

public:

	/// Construct PinnedBuffer with \c n elements
	PinnedBuffer(int n = 0) :
		capacity(0), _size(0), hostptr(nullptr), devptr(nullptr)
	{
		resize_anew(n);
	}

	/// To enable \c std::swap()
	PinnedBuffer (PinnedBuffer&& b)
	{
		*this = std::move(b);
	}

	/// To enable \c std::swap()
	PinnedBuffer& operator=(PinnedBuffer&& b)
	{
		if (this!=&b)
		{
			capacity = b.capacity;
			_size = b._size;
			hostptr = b.hostptr;
			devptr = b.devptr;

			b.capacity = 0;
			b._size = 0;
			b.devptr = nullptr;
			b.hostptr = nullptr;
		}

		return *this;
	}

	/// Release resources and report if debug level is high enough
	~PinnedBuffer()
	{
		if (devptr != nullptr)
		{
			cudaFreeHost(hostptr);
			cudaFree(devptr);
		}
	}

	inline int datatype_size() const { return sizeof(T); }                                      ///< @return sizeof( element )
	inline int size()          const { return _size; }                                          ///< @return number of stored elements

	inline void* genericDevPtr() const { return (void*) devPtr(); }                             ///< @return device pointer void* to the data

	inline void resize     (const int n, cudaStream_t stream) { _resize(n, stream, true);  }    ///< Resize container, don't care about the data. @param n new size, must be >= 0
	inline void resize_anew(const int n)                      { _resize(n, 0,      false); }    ///< Resize container, keep stored data
                                                                                                ///< @param n new size, must be >= 0
	                                                                                            ///< @param stream data will be copied on that CUDA stream


	inline T* hostPtr() const { return hostptr; }  ///< @return typed host pointer to data
	inline T* devPtr()  const { return devptr; }   ///< @return typed device pointer to data

	inline       T& operator[](int i)       { return hostptr[i]; }  ///< allow array-like bracketed access to HOST data
	inline const T& operator[](int i) const { return hostptr[i]; }

	inline T* begin() { return hostptr; }          /// To support range-based loops
	inline T* end()   { return hostptr + _size; }  /// To support range-based loops


	/**
	 * Copy data from device to host
	 *
	 * @param synchronize if false, the call is fully asynchronous.
	 * if true, host data will be readily available on the call return.
	 */
	inline void downloadFromDevice(cudaStream_t stream, bool synchronize = true)
	{
		if (_size > 0) cudaMemcpyAsync(hostptr, devptr, sizeof(T) * _size, cudaMemcpyDeviceToHost, stream);
		if (synchronize) cudaStreamSynchronize(stream);
	}

	/// Copy data from host to device
	inline void uploadToDevice(cudaStream_t stream)
	{
		if (_size > 0) cudaMemcpyAsync(devptr, hostptr, sizeof(T) * _size, cudaMemcpyHostToDevice, stream);
	}

	/// Set all the bytes to 0 on both host and device
	inline void clear(cudaStream_t stream)
	{
		clearDevice(stream);
		clearHost();
	}

	/// Set all the bytes to 0 on device only
	inline void clearDevice(cudaStream_t stream)
	{
		if (_size > 0) cudaMemsetAsync(devptr, 0, sizeof(T) * _size, stream);
	}

	/// Set all the bytes to 0 on host only
	inline void clearHost()
	{
		if (_size > 0) memset(hostptr, 0, sizeof(T) * _size);
	}


	/// Copy data from a PinnedBuffer of the same template type
	void copy(const PinnedBuffer<T>& cont, cudaStream_t stream)
	{
		resize_anew(cont.size());

		if (_size > 0)
		{
			cudaMemcpyAsync(devptr, cont.devPtr(), sizeof(T) * _size, cudaMemcpyDeviceToDevice, stream);
			memcpy(hostptr, cont.hostPtr(), sizeof(T) * _size);
		}
	}
};


