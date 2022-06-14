#ifndef MATH_CORE_POINTERS_H
#define MATH_CORE_POINTERS_H

#include <memory>
#include <cstddef>
#include <compare>

#include <math/core/allocators.h>
#include <math/core/utils.h>

namespace math::core::pointers {
	// These class are not thread safe
	// The behaviour for array, pointer or reference is undefined
	template <typename T, math::core::allocators::Allocator Internal_allocator = math::core::allocators::Malloc_allocator>
	class Unique_ptr {
	public:
		// Not recommended - ptr should be allocated using Internal_allocator
		Unique_ptr(T* ptr = nullptr)
			: ptr_(ptr) {}

		template <typename T_o>
		Unique_ptr(const Unique_ptr<T_o, Internal_allocator>& other) noexcept = delete;
		Unique_ptr(const Unique_ptr& other) noexcept = delete;

		template <typename T_o>
		Unique_ptr& operator=(const Unique_ptr<T_o, Internal_allocator>& other) noexcept = delete;
		Unique_ptr& operator=(const Unique_ptr& other) noexcept = delete;

		template <typename T_o>
		Unique_ptr(Unique_ptr<T_o, Internal_allocator>&& other) noexcept
			: allocator_(other.allocator_), ptr_(other.ptr_)
		{
			other.ptr_ = nullptr;
		}
		Unique_ptr(Unique_ptr&& other) noexcept
			: allocator_(other.allocator_), ptr_(other.ptr_)
		{
			other.ptr_ = nullptr;
		}

		template <typename T_o>
		Unique_ptr& operator=(Unique_ptr<T_o, Internal_allocator>&& other) noexcept
		{
			if (this == &other) {
				return *this;
			}

			remove_reference();

			allocator_ = other.allocator_;
			ptr_ = other.ptr_;

			other.ptr_ = nullptr;
			return *this;
		}
		Unique_ptr& operator=(Unique_ptr&& other) noexcept
		{
			if (this == &other) {
				return *this;
			}

			remove_reference();

			allocator_ = other.allocator_;
			ptr_ = other.ptr_;

			other.ptr_ = nullptr;
			return *this;
		}

		virtual ~Unique_ptr() noexcept
		{
			remove_reference();
		}

		T* get() const noexcept
		{
			return ptr_;
		}

		T* operator->() const noexcept
		{
			return ptr_;
		}

		T& operator*() const noexcept
		{
			return *(ptr_);
		}

		operator bool() const noexcept
		{
			return ptr_;
		}

		void reset() noexcept
		{
			remove_reference();
			ptr_ = nullptr;
		}

		template <typename T_o>
		void reset(T_o* ptr) noexcept
		{
			if (!ptr) {
				reset();
			}
			remove_reference();
			ptr_ = ptr;
		}

		template <typename T_o, math::core::allocators::Allocator Internal_allocator_o>
		friend class Unique_ptr;

		template <typename T_o, math::core::allocators::Allocator Internal_allocator_o>
		friend bool operator==(const Unique_ptr<T_o, Internal_allocator_o>& lhs, const Unique_ptr<T_o, Internal_allocator_o>& rhs);

		template <typename T_o, math::core::allocators::Allocator Internal_allocator_o>
		friend std::strong_ordering operator<=>(const Unique_ptr<T_o, Internal_allocator_o>& lhs, const Unique_ptr<T_o, Internal_allocator_o>& rhs);

		template <typename T_o, math::core::allocators::Allocator Internal_allocator_o>
		friend bool operator==(const Unique_ptr<T_o, Internal_allocator_o>& lhs, std::nullptr_t);

		template <typename T_o, math::core::allocators::Allocator Internal_allocator_o>
		friend std::strong_ordering operator<=>(const Unique_ptr<T_o, Internal_allocator_o>& lhs, std::nullptr_t);

	private:
		void remove_reference()
		{
			// Check if there's an object in use
			if (ptr_) {
				math::core::memory::aux::destruct_at<T>(ptr_);
				math::core::memory::Block ptr_b = { const_cast<std::remove_const_t<T>*>(ptr_), sizeof(T) };
				allocator_.deallocate(&ptr_b);
				ptr_ = nullptr;
			}
		}

		Internal_allocator allocator_{};
		T* ptr_{ nullptr };
	};

	template <typename T, math::core::allocators::Allocator Internal_allocator>
	inline bool operator==(const Unique_ptr<T, Internal_allocator>& lhs, const Unique_ptr<T, Internal_allocator>& rhs)
	{
		return lhs.ptr_ == rhs.ptr_;
	}

	template <typename T, math::core::allocators::Allocator Internal_allocator>
	inline std::strong_ordering operator<=>(const Unique_ptr<T, Internal_allocator>& lhs, const Unique_ptr<T, Internal_allocator>& rhs)
	{
		return std::compare_three_way{}(lhs.ptr_, rhs.ptr_);
	}

	template <typename T, math::core::allocators::Allocator Internal_allocator>
	inline bool operator==(const Unique_ptr<T, Internal_allocator>& lhs, std::nullptr_t)
	{
		return !lhs;
	}

	template <typename T, math::core::allocators::Allocator Internal_allocator>
	inline std::strong_ordering operator<=>(const Unique_ptr<T, Internal_allocator>& lhs, std::nullptr_t)
	{
		return std::compare_three_way{}(lhs.ptr_, nullptr);
	}


	template <typename T, math::core::allocators::Allocator Internal_allocator = math::core::allocators::Malloc_allocator, typename ...Args>
	inline Unique_ptr<T, Internal_allocator> make_unique(Args&&... args)
	{
		Internal_allocator allocator_{};
		math::core::memory::Block b = allocator_.allocate(sizeof(T));
		T* ptr = math::core::memory::aux::construct_at<T>(reinterpret_cast<T*>(b.p), std::forward<Args>(args)...);
		return Unique_ptr<T, Internal_allocator>(ptr);
	}

	template <typename T, math::core::allocators::Allocator Internal_allocator = math::core::allocators::Malloc_allocator>
	class Shared_ptr {
	public:
		// Not recommended - ptr should be allocated using Internal_allocator
		Shared_ptr(T* ptr = nullptr)
			: use_count_(ptr ? reinterpret_cast<std::size_t*>(allocator_.allocate(sizeof(std::size_t)).p) : nullptr), ptr_(ptr)
		{
			CORE_EXPECT((ptr_ && use_count_) || (!ptr_ && !use_count_), std::runtime_error, "internal memory allocation failed");
			if (use_count_) {
				*use_count_ = 1;
			}
		}

		template <typename T_o>
		Shared_ptr(const Shared_ptr<T_o, Internal_allocator>& other) noexcept
			: allocator_(other.allocator_), use_count_(other.use_count_), ptr_(other.ptr_)
		{
			if (other.ptr_ && use_count_) {
				++(*use_count_);
			}
		}
		Shared_ptr(const Shared_ptr& other) noexcept
			: allocator_(other.allocator_), use_count_(other.use_count_), ptr_(other.ptr_)
		{
			if (other.ptr_ && use_count_) {
				++(*use_count_);
			}
		}

		template <typename T_o>
		Shared_ptr(const Shared_ptr<T_o, Internal_allocator>& other, T* ptr) noexcept
			: allocator_(other.allocator_), use_count_(other.use_count_), ptr_(ptr)
		{
			if (other.ptr_ && use_count_) {
				++(*use_count_);
			}
		}

		template <typename T_o>
		Shared_ptr& operator=(const Shared_ptr<T_o, Internal_allocator>& other) noexcept
		{
			if (this == &other) {
				return *this;
			}

			remove_reference();

			allocator_ = other.allocator_;
			use_count_ = other.use_count_;
			ptr_ = other.ptr_;

			if (other.ptr_) {
				++(*use_count_);
			}
			return *this;
		}
		Shared_ptr& operator=(const Shared_ptr& other) noexcept
		{
			if (this == &other) {
				return *this;
			}

			remove_reference();

			allocator_ = other.allocator_;
			use_count_ = other.use_count_;
			ptr_ = other.ptr_;

			if (other.ptr_) {
				++(*use_count_);
			}
			return *this;
		}

		template <typename T_o>
		Shared_ptr(Shared_ptr<T_o, Internal_allocator>&& other) noexcept
			: allocator_(other.allocator_), use_count_(other.use_count_), ptr_(other.ptr_)
		{
			other.use_count_ = nullptr;
			other.ptr_ = nullptr;
		}
		Shared_ptr(Shared_ptr&& other) noexcept
			: allocator_(other.allocator_), use_count_(other.use_count_), ptr_(other.ptr_)
		{
			other.use_count_ = nullptr;
			other.ptr_ = nullptr;
		}

		template <typename T_o>
		Shared_ptr(Shared_ptr<T_o, Internal_allocator>&& other, T* ptr) noexcept
			: allocator_(other.allocator_), use_count_(other.use_count_), ptr_(ptr)
		{
			other.use_count_ = nullptr;
			other.ptr_ = nullptr;
		}

		template <typename T_o>
		Shared_ptr& operator=(Shared_ptr<T_o, Internal_allocator>&& other) noexcept
		{
			if (this == &other) {
				return *this;
			}

			remove_reference();

			allocator_ = other.allocator_;
			use_count_ = other.use_count_;
			ptr_ = other.ptr_;

			other.use_count_ = nullptr;
			other.ptr_ = nullptr;
			return *this;
		}
		Shared_ptr& operator=(Shared_ptr&& other) noexcept
		{
			if (this == &other) {
				return *this;
			}

			remove_reference();

			allocator_ = other.allocator_;
			use_count_ = other.use_count_;
			ptr_ = other.ptr_;

			other.use_count_ = nullptr;
			other.ptr_ = nullptr;
			return *this;
		}

		virtual ~Shared_ptr() noexcept
		{
			remove_reference();
		}

		std::size_t use_count() const noexcept
		{
			return use_count_ ? *use_count_ : 0;
		}

		T* get() const noexcept
		{
			return ptr_;
		}

		T* operator->() const noexcept
		{
			return ptr_;
		}

		T& operator*() const noexcept
		{
			return *(ptr_);
		}

		operator bool() const noexcept
		{
			return ptr_;
		}

		void reset() noexcept
		{
			remove_reference();
			use_count_ = nullptr;
			ptr_ = nullptr;
		}

		template <typename T_o>
		void reset(T_o* ptr) noexcept
		{
			if (!ptr) {
				reset();
			}
			remove_reference();
			use_count_ = reinterpret_cast<std::size_t*>(allocator_.allocate(sizeof(std::size_t)).p);
			*(use_count_) = 1;
			ptr_ = ptr;
        }

		template <typename T_o, math::core::allocators::Allocator Internal_allocator_o>
		friend class Shared_ptr;

		template <typename T_o, math::core::allocators::Allocator Internal_allocator_o>
		friend bool operator==(const Shared_ptr<T_o, Internal_allocator_o>& lhs, const Shared_ptr<T_o, Internal_allocator_o>& rhs);

		template <typename T_o, math::core::allocators::Allocator Internal_allocator_o>
		friend std::strong_ordering operator<=>(const Shared_ptr<T_o, Internal_allocator_o>& lhs, const Shared_ptr<T_o, Internal_allocator_o>& rhs);

		template <typename T_o, math::core::allocators::Allocator Internal_allocator_o>
		friend bool operator==(const Shared_ptr<T_o, Internal_allocator_o>& lhs, std::nullptr_t);

		template <typename T_o, math::core::allocators::Allocator Internal_allocator_o>
		friend std::strong_ordering operator<=>(const Shared_ptr<T_o, Internal_allocator_o>& lhs, std::nullptr_t);

    private:
		void remove_reference()
		{
			// Check if there's an object in use
			if (ptr_ && use_count_) {
				--(*use_count_); // Allocated use count must be positive
				if (*use_count_ == 0) {
					math::core::memory::aux::destruct_at<std::size_t>(use_count_);
					math::core::memory::Block use_count_b = { use_count_, sizeof(std::size_t) };
					allocator_.deallocate(&use_count_b);
					use_count_ = nullptr;

					math::core::memory::aux::destruct_at<T>(ptr_);
					math::core::memory::Block ptr_b = { const_cast<std::remove_const_t<T>*>(ptr_), sizeof(T) };
					allocator_.deallocate(&ptr_b);
					ptr_ = nullptr;
				}
			}
		}

        Internal_allocator allocator_{};
		std::size_t* use_count_{nullptr};
		T* ptr_{nullptr};
    };

	template <typename T, math::core::allocators::Allocator Internal_allocator>
	inline bool operator==(const Shared_ptr<T, Internal_allocator>& lhs, const Shared_ptr<T, Internal_allocator>& rhs)
	{
		return lhs.ptr_ == rhs.ptr_;
	}

	template <typename T, math::core::allocators::Allocator Internal_allocator>
	inline std::strong_ordering operator<=>(const Shared_ptr<T, Internal_allocator>& lhs, const Shared_ptr<T, Internal_allocator>& rhs)
	{
		return std::compare_three_way{}(lhs.ptr_, rhs.ptr_);
	}

	template <typename T, math::core::allocators::Allocator Internal_allocator>
	inline bool operator==(const Shared_ptr<T, Internal_allocator>& lhs, std::nullptr_t)
	{
		return !lhs;
	}

	template <typename T, math::core::allocators::Allocator Internal_allocator>
	inline std::strong_ordering operator<=>(const Shared_ptr<T, Internal_allocator>& lhs, std::nullptr_t)
	{
		return std::compare_three_way{}(lhs.ptr_, nullptr);
	}


	template <typename T, math::core::allocators::Allocator Internal_allocator = math::core::allocators::Malloc_allocator, typename ...Args>
	inline Shared_ptr<T, Internal_allocator> make_shared(Args&&... args)
	{
		Internal_allocator allocator_{};
		math::core::memory::Block b = allocator_.allocate(sizeof(T));
		T* ptr = math::core::memory::aux::construct_at<T>(reinterpret_cast<T*>(b.p), std::forward<Args>(args)...);
		return Shared_ptr<T, Internal_allocator>(ptr);
	}

	template <typename T, typename U, math::core::allocators::Allocator Internal_allocator = math::core::allocators::Malloc_allocator>
	inline Shared_ptr<T, Internal_allocator> static_pointer_cast(const Shared_ptr<U, Internal_allocator>& other) noexcept
	{
		T* p = static_cast<T*>(other.get());
		return Shared_ptr<T, Internal_allocator>(other, p);
	}

	template <typename T, typename U, math::core::allocators::Allocator Internal_allocator = math::core::allocators::Malloc_allocator>
	inline Shared_ptr<T, Internal_allocator> static_pointer_cast(Shared_ptr<U, Internal_allocator>&& other) noexcept
	{
		T* p = static_cast<T*>(other.get());
		return Shared_ptr<T, Internal_allocator>(std::move(other), p);
	}

	template <typename T, typename U, math::core::allocators::Allocator Internal_allocator = math::core::allocators::Malloc_allocator>
	inline Shared_ptr<T, Internal_allocator> dynamic_pointer_cast(const Shared_ptr<U, Internal_allocator>& other) noexcept
	{
		if (T* p = dynamic_cast<T*>(other.get())) {
			return Shared_ptr<T, Internal_allocator>(other, p);
		}
		return Shared_ptr<T, Internal_allocator>{};
	}

	template <typename T, typename U, math::core::allocators::Allocator Internal_allocator = math::core::allocators::Malloc_allocator>
	inline Shared_ptr<T, Internal_allocator> dynamic_pointer_cast(Shared_ptr<U, Internal_allocator>&& other) noexcept
	{
		if (T* p = dynamic_cast<T*>(other.get())) {
			return Shared_ptr<T, Internal_allocator>(std::move(other), p);
		}
		return Shared_ptr<T, Internal_allocator>{};
	}

	template <typename T, typename U, math::core::allocators::Allocator Internal_allocator = math::core::allocators::Malloc_allocator>
	inline Shared_ptr<T, Internal_allocator> const_pointer_cast(const Shared_ptr<U, Internal_allocator>& other) noexcept
	{
		T* p = const_cast<T*>(other.get());
		return Shared_ptr<T, Internal_allocator>(other, p);
	}

	template <typename T, typename U, math::core::allocators::Allocator Internal_allocator = math::core::allocators::Malloc_allocator>
	inline Shared_ptr<T, Internal_allocator> const_pointer_cast(Shared_ptr<U, Internal_allocator>&& other) noexcept
	{
		T* p = const_cast<T*>(other.get());
		return Shared_ptr<T, Internal_allocator>(std::move(other), p);
	}

	template <typename T, typename U, math::core::allocators::Allocator Internal_allocator = math::core::allocators::Malloc_allocator>
	inline Shared_ptr<T, Internal_allocator> reinterpret_pointer_cast(const Shared_ptr<U, Internal_allocator>& other) noexcept
	{
		T* p = reinterpret_cast<T*>(other.get());
		return Shared_ptr<T, Internal_allocator>(other, p);
	}

	template <typename T, typename U, math::core::allocators::Allocator Internal_allocator = math::core::allocators::Malloc_allocator>
	inline Shared_ptr<T, Internal_allocator> reinterpret_pointer_cast(Shared_ptr<U, Internal_allocator>&& other) noexcept
	{
		T* p = reinterpret_cast<T*>(other.get());
		return Shared_ptr<T, Internal_allocator>(std::move(other), p);
	}
}

#endif // MATH_CORE_POINTERS_H

