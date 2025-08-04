# import asyncio
# import sys
#
# from datamin_agent import DataMiningAgent
#
# async def main():
#     user_input = {
#         "姓名": "晨睿睿",
#         "性别": "男",
#         "民族": "汉族",
#         "电话号码": "13908378823",
#         "身份证号": "44160198823940284672123",
#         "籍贯": "杭州市",
#         "出生日期": "2001-01-01",
#         "电子邮箱": "239876102@gmail.com",
#         "职业": "学生",
#         "地址": "广州市",
#         "户口": "北京市",
#         "政治面貌": "群众",
#         "婚姻状况": "已婚",
#         "宗教": "佛教",
#         "学历": "小学"
#     }
#
#     business_id = 0
#     data = DataMiningAgent()
#
#     try:
#         print("测试流式输出：")
#         print(end=' ', flush=True)
#         # 获取生成器对象
#         flow_generator = await data.get_flow(idx=business_id, user_info=user_input)
#
#         # 验证生成器有效性（避免对None或非生成器对象迭代）
#         if not hasattr(flow_generator, "__aiter__"):
#             print("错误：未获取到有效的流式生成器", file=sys.stderr)
#             return
#
#         # 流式接收并打印结果
#         pr = 0
#         async for chunk in flow_generator:
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
#      asyncio.run(main())