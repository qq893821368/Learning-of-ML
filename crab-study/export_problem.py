#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pymysql

# 创建数据库链接
db = pymysql.connect("localhost", "root", "194229lyy", "jol")

# cursor()创建游标
cursor = db.cursor()

try:
    fh = open("y.item", "w")

    # 使用 execute()  方法执行 SQL 查询
    cursor.execute("select problem_id from problem where defunct = 'N'")

    # 使用 fetchall() 方法获取所有结果数据.
    dataset = cursor.fetchall()

    # 遍历输出到文件
    for row in dataset:
        line_string = [str(row[0]), "\n"]
        fh.writelines(line_string)

except IOError:
    print("Error: 读取文件失败")
else:
    print("写入文件成功")
    fh.close()


# 关闭数据库连接
db.close()