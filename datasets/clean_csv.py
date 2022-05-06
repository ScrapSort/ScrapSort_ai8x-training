# import pandas as pd
# import re
# pd.options.mode.chained_assignment = None 

# #df = pd.read_csv("/home/geffen/Desktop/sorting_imgs_all/Bottle.csv")
# df = pd.read_csv("/home/geffen_cooper/ScrapSort/sorting_imgs_all/Other.csv")
# df = df.drop(['file_size','file_attributes','region_count','region_id','region_attributes'],axis=1)
# # df = df.sort_values(by=['filename'])
# #print(df.head)
# df['x'] = 0
# df['y'] = 0
# df['w'] = 0
# df['h'] = 0
# for i in range(len(df)):
#     str = df.iloc[i]['region_shape_attributes']
#     bb = re.findall(r'\d+',str)
#     if len(bb) == 0:
#         print(df.iloc[i])
#         continue
#     df.at[i,'x'] = int(bb[0])
#     df.at[i,'y'] = int(bb[1])
#     df.at[i,'w'] = int(bb[2])
#     df.at[i,'h'] = int(bb[3])

# df = df.drop(['region_shape_attributes'],axis=1)
# df.to_csv("/home/geffen_cooper/ScrapSort/sorting_imgs_all/Other.csv")
# # b = df.loc[df['filename'] == 'img0535.png']['region_shape_attributes'].to_list()
# # print(b[0])