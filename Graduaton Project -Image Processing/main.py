from matplotlib.pyplot import imshow
from main_functions import *


side_score ,errorname_side ,image_f_final,image_c_final = get_score("side")

if side_score > 0.90 :
    print("Exceeded 90% Treshold Value from Side Camera")
    print("Similarity score from Side Camera:",side_score)
else:
    error_compare("side", errorname_side,image_c_final,image_f_final)
    print("Similarity score from Side Camera:", side_score)  
    print("Could NOT Exceeded 90% Treshold Value from Side Camera")
    


front_score ,errorname_front,image_f_final,image_c_final = get_score("front")


if front_score > 0.90 :
    print("Exceeded 90% Treshold Value from Front Camera")
    print("Similarity score from Front Camera:", front_score)
else:
    error_compare("front",errorname_front,image_c_final,image_f_final)
    print("Similarity score from Front Camera:", front_score)
    print("Could NOT Exceeded 90% Treshold Value from Front Camera")

weight1_1 = 0.3
weight2_1 = 1 - weight1_1
final_score = round(side_score * weight1_1 + front_score * weight2_1, 2)
print("Final score from both cameras",final_score)
if(final_score>0.9):
    print("MANUFACTURED OBJECT IS SUITABLE TO USE") 







