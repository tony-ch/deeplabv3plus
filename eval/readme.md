评估模型的相关代码，并不直接使用模型文件，而是根据exp/vis目录下的测试结果计算pixel accuracy和mean IoU。

直接执行的文件是run.sh. run.sh调用eval.py评价模型，根据需要调整run.sh中的列表以完成评估。