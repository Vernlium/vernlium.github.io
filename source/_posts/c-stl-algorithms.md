---
title: c++ stl algorithms
date: 2020-02-23 21:53:47
tags: [c++, stl]
---

# C++ STL algorithms

## 头文件

### <algorithm>,<numeric>,<functional>组成。

### 要使用 STL中的算法函数必须包含头文件<algorithm>

### 对于数值算法须包含<numeric>

### <functional>中则定义了一些模板类，用来声明函数对象

## 4类算法

### 非可变序列算法

- 指不直接修改其所操作的容器内容的算法。

### 可变序列算法

- 指可以修改它们所操作的容器内容的算法。

### 排序算法

- 包括对序列进行排序和合并的算法、搜索算法以及有序序列上的集合操作

### 数值算法

- 对容器内容进行数值计算

## 查找算法

### adjacent_find

- 在iterator对标识元素范围内，查找一对相邻重复元素，找到则返回指向这对元素的第一个元素的ForwardIterator。否则返回last。

###    binary_search

- 在有序序列中查找value，找到返回true。

###    count

- 利用等于操作符，把标志范围内的元素与输入值比较，返回相等元素个数。

###    count_if

- 利用输入的操作符，对标志范围内的元素进行操作，返回结果为true的个数。

###     equal_range

- 功能类似equal，返回一对iterator，第一个表示lower_bound，第二个表示upper_bound。

###    find

- 利用底层元素的等于操作符，对指定范围内的元素与输入值进行比较。当匹配时，结束搜索，返回该元素的一个InputIterator。

###     find_end

- 在指定范围内查找"由输入的另外一对iterator标志的第二个序列"的最后一次出现。找到则返回最后一对的第一个ForwardIterator，否则返回输入的"另外一对"的第一个ForwardIterator。

###   find_first_of

- 在指定范围内查找"由输入的另外一对iterator标志的第二个序列"中任意一个元素的第一次出现。重载版本中使用了用户自定义操作符。

###   find_if

- 使用输入的函数代替等于操作符执行find

###    lower_bound

- 返回一个ForwardIterator，指向在有序序列范围内的可以插入指定值而不破坏容器顺序的第一个位置。

###    upper_bound

- 返回一个ForwardIterator，指向在有序序列范围内插入value而不破坏容器顺序的最后一个位置，该位置标志一个大于value的值。

###    search

- 给出两个范围，返回一个ForwardIterator，查找成功指向第一个范围内第一次出现子序列(第二个范围)的位置，查找失败指向last。

###     search_n

- 在指定范围内查找val出现n次的子序列。

## 排序和通用算法

### inplace_merge

- 合并两个有序序列，结果序列覆盖两端范围。

### merge

- 合并两个有序序列，存放到另一个序列。

### nth_element

- 将范围内的序列重新排序，使所有小于第n个元素的元素都出现在它前面，而大于它的都出现在后面。

### partial_sort

- 对序列做部分排序，被排序元素个数正好可以被放到范围内。

### partial_sort_copy

- 与partial_sort类似，不过将经过排序的序列复制到另一个容器。

### partition

- 对指定范围内元素重新排序，使用输入的函数，把结果为true的元素放在结果为false的元素之前。

### stable_partition

- 与partition类似，不过不保证保留容器中的相对顺序。

### random_shuffle

- 对指定范围内的元素随机调整次序。

### reverse

- 将指定范围内元素重新反序排序。

### reverse_copy

- 与reverse类似，不过将结果写入另一个容器。

### rotate

- 将指定范围内元素移到容器末尾，由middle指向的元素成为容器第一个元素。

### rotate_copy

- 与rotate类似，不过将结果写入另一个容器。

### sort

- 以升序重新排列指定范围内的元素。

### stable_sort

- 与sort类似，不过保留相等元素之间的顺序关系。

## 删除和替换算法

### copy

### copy_backward

### iter_swap

### remove

### remove_copy

### remove_if

### remove_copy

### replace

### replace_copy

### replace_if

### replace_copy

### swap

### swap_range

### unique

### unique_copy

## 排列组合算法

### next_permutation

- 取出当前范围内的排列，并重新排序为下一个排列。重载版本使用自定义的比较操作。

### prev_permutation

-  取出指定范围内的序列并将它重新排序为上一个序列。如果不存在上一个序列则返回false。

## 算术算法

### accumulate

### partial_sum

### inner_product

### adjacent_difference

## 生成和异变算法

### fill

### fill_n

### for_each

### generate

### generate_n

### transform

## 关系算法

### equal

### includes

### lexicographical_compare

### max

### max_element

### min

### min_element

### mismatch

## 集合算法

### set_union

### set_intersection

### set_difference

### set_symmetric_difference

## 堆算法

### make_heap

- 把指定范围内的元素生成一个堆。

### pop_heap

- 并不真正把最大元素从堆中弹出，而是重新排序堆。它把first和last-1交换，然后重新生成一个堆。可使用容器的back来访问被"弹出"的元素或者使用pop_back进行真正的删除。

### push_heap

- 假设first到last-1是一个有效堆，要被加入到堆的元素存放在位置last-1，重新生成堆。在指向该函数前，必须先把元素插入容器后。

### sort_heap

- 对指定范围内的序列重新排序，它假设该序列是个有序堆。

