from json import load,dump
import os
from pathlib import Path

PATH=Path(os.path.abspath(__file__)).parent

with open(PATH / "organized_user_order.json") as f:
    JSON=load(f)

error_json={}
modified_json={}

for user_id in JSON:
    orders=JSON[user_id]
    error_json[user_id]={"size":len(orders),"test":orders[str(len(orders))]["eval_set"]=="test"}
    modified_json[user_id]={}
    for order_num in range(1,len(orders)+1):
        order_num=str(order_num)
        order=orders[order_num]
        if order["eval_set"]=="test":
            error_json[user_id]["size"]-=1
        else:
            del order["eval_set"]
            modified_json[user_id][order_num]=order
with open(PATH / "meta.json", "w") as f:
    dump(error_json,f)
with open(PATH / "modified.json","w") as f:
    dump(modified_json,f)