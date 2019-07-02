#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pymysql

# 创建数据库链接
db = pymysql.connect("localhost", "root", "194229lyy", "jol")

# cursor()创建游标
cursor = db.cursor()

try:
    fh = open("y.data", "w")
    try:
        # 使用 execute()  方法执行 SQL 查询
        cursor.execute("select user_id, problem_id, rating from problem_rating")

        # 使用 fetchall() 方法获取所有结果数据.
        dataset = cursor.fetchall()

        # 遍历输出到文件
        for row in dataset:
            line_string = [str(row[0]), "\t", str(row[1]), "\t", str(row[2]), "\n"]
            fh.writelines(line_string)

        line_string = ["15201211", "\t", "1002", "\t", "5.0", "\n"]
        fh.writelines(line_string)

        line_string = ["15201211", "\t", "1003", "\t", "3.0", "\n"]
        fh.writelines(line_string)

        line_string = ["15201211", "\t", "1004", "\t", "4.0", "\n"]
        fh.writelines(line_string)

        line_string = ["15201210", "\t", "1002", "\t", "5.0", "\n"]
        fh.writelines(line_string)

        line_string = ["15201210", "\t", "1003", "\t", "1.0", "\n"]
        fh.writelines(line_string)
    except:
        print('error')
except IOError:
    print("Error: 读取文件失败")
else:
    print("写入文件成功")
    fh.close()


# 关闭数据库连接
db.close()