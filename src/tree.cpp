#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/call.hpp>

#include <boost/math/constants/constants.hpp>

#include <iostream>

#include <algorithm>
#include <deque>
#include <forward_list>
#include <iterator>
#include <numeric>
#include <optional>
#include <unordered_map>
#include <variant>
#include <vector>

namespace p  = boost::python;
namespace np = boost::python::numpy;


class random_cache {
public:
	typedef std::ptrdiff_t difference_type;
	typedef double value_type;
	typedef value_type* pointer;
	typedef value_type& reference;
	typedef std::input_iterator_tag iterator_category;

	random_cache(const p::object& uniform_call, std::size_t size = 1024):
		uniform_call_(uniform_call),
		size_(size),
		cache_(refill()),
		pos_(0) {

		assert(np::dtype::get_builtin<value_type>() == cache_.get_dtype());
	}

	random_cache(const random_cache&) = default;
	random_cache(random_cache&&) = default;
	random_cache& operator=(const random_cache&) = default;
	random_cache& operator=(random_cache&&) = default;

	random_cache& operator++() {
		rewind();

		return *this;
	}

	random_cache operator++(int) {
		auto it = *this;

		rewind();

		return it;
	}

	const reference operator*() const {
		return reinterpret_cast<pointer>(cache_.get_data())[pos_];
	}

	bool operator==(const random_cache& other) const {
		return uniform_call_ == other.uniform_call_ && size_ == other.size_ && cache_ == other.cache_ && pos_ == other.pos_;
	}
	bool operator!=(const random_cache& other) const {
		return !(*this == other);
	}
private:
	void rewind() {
		if (++pos_ == size_) {
			cache_ = refill();
			pos_ = 0;
		}
	}

	np::ndarray refill() const {
		return p::call<np::ndarray>(uniform_call_.ptr(), 0.0, 1.0, size_);
	}

private:
	p::object uniform_call_;
	const std::size_t size_;
	np::ndarray cache_;
	std::size_t pos_;
};


class tree {
private:
	struct empty;
	struct split;
	struct leaf;
	using node = std::variant<empty, split, leaf>;
public:
	tree(std::size_t max_depth, std::size_t max_iter, const p::object& random_generator):
		max_depth_(max_depth),
		max_iter_(max_iter),
		random_(random_generator.attr("uniform")) {
	}

	void fit(const p::object& X) {
		const auto& data = p::extract<np::ndarray>(X.attr("data"))();
		const auto& mask = p::extract<np::ndarray>(X.attr("mask"))();

		fit(data, mask);
	}

	np::ndarray predict(const p::object& X) const {
		const auto& data = p::extract<np::ndarray>(X.attr("data"))();
		const auto& mask = p::extract<np::ndarray>(X.attr("mask"))();

		return predict(data, mask);
	}
private:
	void fit(const np::ndarray& data, const np::ndarray& mask);
	np::ndarray predict(const np::ndarray& data, const np::ndarray& mask) const;
	float predict_one(const double* data, const std::size_t data_stride, const bool* mask, const std::size_t mask_stride) const;
public:
	const std::size_t max_depth_;
	const std::size_t max_iter_;
	random_cache random_;
	std::deque<node> nodes_;
	std::vector<float> leaf_density_;
};

struct tree::empty {
};

struct tree::split {
	std::size_t left;
	std::size_t right; // redundant?
	std::size_t feature;
	double value;
};

struct tree::leaf {
	std::size_t id;
	std::size_t size;
};

void tree::fit(const np::ndarray& data, const np::ndarray& mask) {
	struct frame {
		using list_type = std::forward_list<std::size_t>;

		std::size_t id;
		std::size_t depth;
		list_type objects;

		frame(std::size_t id, std::size_t depth, list_type&& objects):
			id(id), depth(depth), objects(std::move(objects)) {
		}
		frame(frame&&) = default;
		frame(const frame&) = delete;
		frame& operator=(frame&&) = default;
		frame& operator=(const frame&) = delete;


		std::optional<std::pair<std::size_t, std::pair<double, double>>> select_split_feature(const np::ndarray& data, const np::ndarray& mask, random_cache& random) const {
			std::vector<std::size_t> count(data.shape(1), 0);
			std::vector<double> max_value(data.shape(1), -std::numeric_limits<double>::infinity());
			std::vector<double> min_value(data.shape(1), std::numeric_limits<double>::infinity());

			const auto row_stride = data.strides(0) / sizeof(double);
			const auto col_stride = data.strides(1) / sizeof(double);
			const auto mask_row_stride = mask.strides(0) / sizeof(bool);
			const auto mask_col_stride = mask.strides(1) / sizeof(bool);

			const auto pdata = reinterpret_cast<const double*>(data.get_data());
			const auto pmask = reinterpret_cast<const bool*>(mask.get_data());

			for (const auto& i: objects) {
				auto col_iter = pdata + i * row_stride;
				auto mask_col_iter = pmask + i * mask_row_stride;
				std::size_t j = 0;
				for (; j < data.shape(1); ++j, col_iter += col_stride, mask_col_iter += mask_col_stride) {
					if (!*mask_col_iter) {
						count[j]++;
						max_value[j] = std::max(max_value[j], *col_iter);
						min_value[j] = std::min(min_value[j], *col_iter);
					}
				}
			}

			const auto max_count = std::max_element(count.cbegin(), count.cend());

			std::deque<std::vector<std::size_t>::const_iterator> all_max;
			for (auto it = count.cbegin(); it != count.cend(); ++it) {
				if (*it == *max_count) {
					all_max.push_back(it);
				}
			}

			const auto split_count = *(all_max.begin() + std::size_t((*++random) * all_max.size()));

			if (*split_count > 1) {
				const auto feature = std::distance(count.cbegin(), split_count);
				const auto min_max = std::make_pair(min_value[feature], max_value[feature]);

				return std::make_pair(feature, min_max);
			}

			return {};
		}

		std::optional<std::tuple<std::size_t, double, double>> select_split(const np::ndarray& data, const np::ndarray& mask, random_cache& random) const noexcept {
			const auto maybe_feature = select_split_feature(data, mask, random);

			if (!maybe_feature) return {};

			const auto [feature, min_max] = *maybe_feature;
			const auto [min_value, max_value] = min_max;
			const auto raw_value = (*++random);
			const auto value = min_value + (max_value - min_value) * raw_value;

			return std::make_tuple(feature, value, raw_value);
		};

		static std::pair<list_type, list_type> split(const np::ndarray& data, const np::ndarray& mask, std::size_t feature, double value, list_type objects) noexcept {
			list_type left;

			const auto row_stride = data.strides(0) / sizeof(double);
			const auto col_stride = data.strides(1) / sizeof(double);
			const auto mask_row_stride = mask.strides(0) / sizeof(bool);
			const auto mask_col_stride = mask.strides(1) / sizeof(bool);

			const auto pdata = reinterpret_cast<const double*>(data.get_data()) + feature * col_stride;
			const auto pmask = reinterpret_cast<const bool*>(mask.get_data()) + feature * mask_col_stride;

			auto pre_it = objects.before_begin();
			auto it = objects.begin();
			while (it != objects.end()) {
				auto col_iter = pdata + *it * row_stride;
				auto mask_col_iter = pmask + *it * mask_row_stride;

				if (*mask_col_iter) { // both
					left.emplace_front(*it);
					pre_it = it++;
				} else if (*col_iter < value) {
					left.splice_after(left.before_begin(), objects, pre_it);
					it = std::next(pre_it); // repair it
				} else {
					pre_it = it++;
				}
			}

			return std::make_pair(std::move(left), std::move(objects));
		}
	};

	std::size_t next_leaf_id = 0;
	std::unordered_multimap<std::size_t, std::size_t> index; /* object_id -> leaf_id */
	std::forward_list<frame> stack;

	/* Prepare root node */
	{
		frame::list_type all_objects;
		for (auto i = 0; i < data.shape(0); ++i) {
			all_objects.emplace_front(i);
		}
		stack.emplace_front(0, 0, std::move(all_objects));
		nodes_.emplace_back(empty{});
	}

	while (!stack.empty()) {
		auto c = std::move(stack.front());
		stack.pop_front();

		if (c.depth < max_depth_) {
			if (auto s = c.select_split(data, mask, random_); s) {
				auto [feature, value, raw_value] = *s;
				auto [lo, ro] = frame::split(data, mask, feature, value, std::move(c.objects));

				/* Prepare left */
				const auto left_id = nodes_.size();
				stack.emplace_front(left_id, c.depth + 1, std::move(lo));
				nodes_.emplace_back(empty{});

				/* Prepare right */
				const auto right_id = nodes_.size();
				stack.emplace_front(right_id, c.depth + 1, std::move(ro));
				nodes_.emplace_back(empty{});

				nodes_[c.id] = split{left_id, right_id, feature, value};
				continue;
			}
		}

		/* Make leaf */
		std::size_t leaf_size = 0;
		for (const auto& o : c.objects) {
			index.emplace(o, next_leaf_id);
			++leaf_size;
		}
		nodes_[c.id] = leaf{next_leaf_id++, leaf_size};
	}

	/* Init leaf weight */
	std::vector<float> tau(next_leaf_id, static_cast<float>(1)/next_leaf_id);
	std::vector<float> tm(index.size());

	for (std::size_t iter = 0; iter < max_iter_; ++iter) {
		/* Unroll T */
		auto prev_it = index.cbegin();
		auto prev_tm_it = tm.begin();
		float accum = 0;
		const auto normalize = [&accum = std::as_const(accum)] (auto& x) { x /= accum; };
		for (auto [it, tm_it] = std::make_pair(index.cbegin(), tm.begin());
			it != index.cend() && tm_it != tm.end();
			++it, ++tm_it) {

			if (it->first != prev_it->first) {
				std::for_each(prev_tm_it, tm_it, normalize);
				accum = 0;
				prev_it = it;
				prev_tm_it = tm_it;
			}

			accum += (*tm_it = tau[it->second]);
		}
		std::for_each(prev_tm_it, tm.end(), normalize);

		/* Collect tau */
		std::fill(tau.begin(), tau.end(), 0);
		for (auto [it, tm_it] = std::make_pair(index.cbegin(), tm.cbegin());
			it != index.cend() && tm_it != tm.cend();
			++it, ++tm_it) {

			tau[it->second] += *tm_it;
		}

		/* Norm tau */
		const auto norm = std::accumulate(tau.cbegin(), tau.cend(), static_cast<float>(0));
		std::for_each(tau.begin(), tau.end(), [&norm] (auto& x) { x /= norm; });
	}

	leaf_density_ = std::move(tau);
}

np::ndarray tree::predict(const np::ndarray& data, const np::ndarray& mask) const {
	const auto shape = p::make_tuple(data.shape(0), 1);
	auto ret = np::zeros(shape, np::dtype::get_builtin<float>());

	const auto row_stride = data.strides(0) / sizeof(double);
	const auto col_stride = data.strides(1) / sizeof(double);
	const auto mask_row_stride = mask.strides(0) / sizeof(bool);
	const auto mask_col_stride = mask.strides(1) / sizeof(bool);
	const auto ret_stride = ret.strides(0) / sizeof(float);

	const auto pdata = reinterpret_cast<const double*>(data.get_data());
	const auto pmask = reinterpret_cast<const bool*>(mask.get_data());
	auto pret = reinterpret_cast<float*>(ret.get_data());

	auto row_iter = pdata;
	auto mask_row_iter = pmask;
	auto ret_iter = pret;
	std::size_t i = 0;
	for (; i < data.shape(0); ++i, row_iter += row_stride, mask_row_iter += mask_row_stride, ret_iter += ret_stride) {
		*ret_iter = predict_one(row_iter, col_stride, mask_row_iter, mask_col_stride);
	}

	return ret;
}

float tree::predict_one(const double* data, const std::size_t data_stride, const bool* mask, const std::size_t mask_stride) const {
	struct frame {
		std::size_t id;
		std::size_t depth;

		explicit frame(std::size_t id, std::size_t depth):
			id(id), depth(depth) {
		}
		frame(frame&&) = default;
		frame(const frame&) = delete;
		frame& operator=(frame&&) = default;
		frame& operator=(const frame&) = delete;
	};

	float depth_sum = 0;
	float norm = 0;

	std::forward_list<frame> stack;
	stack.emplace_front(std::size_t(0), std::size_t(1));

	while (!stack.empty()) {
		auto c = std::move(stack.front());
		stack.pop_front();

		const auto& node = nodes_[c.id];
		std::visit([&, this](const auto& x) {
			using T = std::decay_t<decltype(x)>;

			if constexpr (std::is_same_v<T, split>) {
				const auto pmask = mask + x.feature * mask_stride;
				const auto pdata = data + x.feature * data_stride;
				const auto newdepth = c.depth + 1;

				if (*pmask) { // both
					stack.emplace_front(x.left, newdepth);
					stack.emplace_front(x.right, newdepth);
				} else if (*pdata < x.value) {
					stack.emplace_front(x.left, newdepth);
				} else {
					stack.emplace_front(x.right, newdepth);
				}
			} else if constexpr (std::is_same_v<T, leaf>) {
				const auto fdepth = static_cast<float>(c.depth);
				const auto depth = (x.size == 1 ? fdepth : fdepth + 2*(boost::math::constants::euler<float>()-1+std::log(static_cast<float>(x.size))));
				const auto tau = leaf_density_[x.id] / x.size;

				depth_sum += tau * depth;
				norm += tau;
			} else {
				assert(false);
			}
		}, node);
	}

	return depth_sum / norm;
}

BOOST_PYTHON_MODULE(tree)
{
	using namespace boost::python;
	using namespace boost::python::numpy;

	numpy::initialize();

	void (tree::*fit)(const object&) = &tree::fit;
	ndarray (tree::*predict)(const object&) const = &tree::predict;

	class_<tree>("Tree", init<std::size_t, std::size_t, object>())
		.def_readonly("max_depth", &tree::max_depth_)
		.def_readonly("max_iter", &tree::max_iter_)
		.def("fit", fit)
		.def("predict", predict);
}

