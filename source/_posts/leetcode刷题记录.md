---
title: leetcode刷题记录
date: 2020-02-23 21:52:29
tags: [leetcode]
---

记录最近一段时间leetcode刷题的总结。

## 1. 两数之和

```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        vector<int> res;
        map<int, int> values;
        map<int, int>::iterator iter;
        for (int i = 0; i < nums.size(); i++) {
            iter = values.find(target - nums[i]);
            if (iter != values.end()) {
                res.push_back(iter->second);
                res.push_back(i);
                return res;
            }
            values[nums[i]] = i;
        }
    }
};
```

编译时，报如下错误信息：

```
solution.cpp: In member function twoSum
Line 5: Char 23: error: control reaches end of non-void function [-Werror=return-type]
         map<int, int> values;
                       ^~~~~~
cc1plus: some warnings being treated as errors
```

原因是，**函数没有返回值，要把`return res;`放到函数的最后。**
