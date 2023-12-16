# DCT-watermarker
课程 网络信息安全与信息隐藏 大作业实现代码
## environment
|     Item      | Detail |
| :-----------: | :----: |
|    python     | 3.9.16 |
| opencv-python | 4.8.1  |
|    mpi4py     | 3.1.5  |
| cryptography  | 41.0.7 |

## parameter  setting
### 1. 
parameter`key` and `r` in `logistic_map_encryption`.
```python
logistic_map_encryption(image, key, r=3.99)
```
## 2.
`key = b"Wat3rmark3r1sFun"` for AES-128bit ECB encryption and decryption.
## 3.
`\alpha` for intensity of adding  watermarker.  default setting: `\alpha=0.1`
