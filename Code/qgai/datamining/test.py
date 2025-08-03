# import asyncio
# import sys
#
# from datamin_agent import DataMiningAgent
#
# async def main():
#     user_input = {
#         "姓名": "陆潇峰",
#         "年龄": 19,
#         "性别": "不愿透露",
#         "民族": "汉",
#         "联系电话": "14111451541",
#         "籍贯": "肇庆",
#         "政治面貌": "群众",
#         "宗教信仰": "无",
#         "婚姻状况": "未婚",
#         "出生日期": "20060606",
#         "居住地": "广州",
#         "户口": "肇庆",
#     }
#
#     business_id = 0
#     data = DataMiningAgent()
#
#     try:
#         print("测试流式输出：")
#         print(end=' ', flush=True)
#
#         # 流式接收并打印结果
#         pr = 0
#         async for chunk in data.get_flow(idx=business_id, user_info=user_input):
#             if "--" in chunk:
#                 pr = 1
#             if pr == 1:
#                 print(chunk, end='', flush=True)
#                 sys.stdout.flush()  # 确保内容即时显示
#
#         print()  # 最后换行
#         pr = 0
#
#     except Exception as e:
#         print(f"推理出错：{e}")
#
# # 测试流式输出
# if __name__ == "__main__":
#      asyncio.new_event_loop().run_until_complete(main())