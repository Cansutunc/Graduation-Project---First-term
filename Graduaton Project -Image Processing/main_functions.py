import operator
from feature_extractor import *

similarity_threshold = 0.90
weight1 = 0.5
weight2 = 1 - weight1

def side_input_images():
    image_c = cv2.imread(r"side_input_images/1/1_c1.jpg", 0)
    image_f = cv2.imread(r"side_input_images/1/1_f1.jpg", 0)

    compare1 = cv2.imread(r"side_input_images/1/1_c1.jpg", 0)
    compare2 = cv2.imread(r"side_input_images/4/4_c1.jpg", 0)
    compare3 = cv2.imread(r"side_input_images/5/5_c1.jpg", 0)



    return image_c,image_f,compare1,compare2,compare3


def front_input_images():
    image_c = cv2.imread(r"input_images/1/1_c.jpg", 0)
    image_f = cv2.imread(r"input_images/1/1_c5.jpg", 0)

    compare1 = cv2.imread(r"input_images/1/1_c.jpg", 0)
    compare2 = cv2.imread(r"input_images/4/4_c.jpg", 0)
    compare3 = cv2.imread(r"input_images/5/5_c.jpg", 0)

    image_c = image_c[100:370,:]
    image_f = image_f[100:370,:]
    compare1 = compare1[100:370, :]
    compare2 = compare2[100:370, :]
    compare3 = compare3[100:370, :]

    return image_c,image_f,compare1,compare2,compare3


def get_extract_countours(type):
    if type=="side":
        side_c, side_f, comp_side1, comp_side2, comp_side3 = side_input_images()
        image_c_final, c_contours = extract_contours(side_c)
        image_f_final, f_contours = extract_contours(side_f)
        comp1_final, f1_contours = extract_contours(comp_side1)
        comp2_final, f2_contours = extract_contours(comp_side2)
        comp3_final, f3_contours = extract_contours(comp_side3)
        return image_c_final,c_contours,image_f_final,f_contours,comp1_final,f1_contours,comp2_final,f2_contours,comp3_final,f3_contours
    elif type=="front":
        front_c, front_f, comp_front1, comp_front2, comp_front3 = front_input_images()
        image_c_final, c_contours = extract_contours(front_c)
        image_f_final, f_contours = extract_contours(front_f)
        comp1_final, f1_contours = extract_contours(comp_front1)
        comp2_final, f2_contours = extract_contours(comp_front2)
        comp3_final, f3_contours = extract_contours(comp_front3)
        return image_c_final, c_contours, image_f_final, f_contours, comp1_final, f1_contours, comp2_final, f2_contours, comp3_final, f3_contours


def get_score(type):

    if type=="side":
        side_image_c_final, side_c_contours, side_image_f_final, side_f_contours, side_comp1_final, side_f1_contours, side_comp2_final, side_f2_contours, side_comp3_final, side_f3_contours = get_extract_countours(
            "side")
        if side_c_contours is not None or side_f_contours is not None:
            # cv2.imshow("image_c_final", side_image_c_final)
            # cv2.imshow("image_f_final", side_image_f_final)

            result, score = compare_binary_images(side_image_c_final, side_image_f_final, similarity_threshold)

            error_name = compare3_images(side_image_c_final, side_comp1_final, side_comp2_final, side_comp3_final)
            # print(error_name)
            result2, score2, score_scaled2 = compare_contours(side_c_contours, side_f_contours, similarity_threshold)
            # print(score2)
            # print(score_scaled2)

            final_score = round(score * weight1 + score_scaled2 * weight2, 2)
            # print(str(similarity_threshold <= final_score) + "-> Final Score: " + str(final_score),"type ="+type)
            return final_score ,error_name,side_image_f_final ,side_image_c_final
    elif type=="front":
        front_image_c_final, front_c_contours, front_image_f_final, front_f_contours, front_comp1_final, front_f1_contours, front_comp2_final, front_f2_contours, front_comp3_final, front_f3_contours = get_extract_countours(
            "front")
        if front_c_contours is not None or front_f_contours is not None:
            # cv2.imshow("image_c_final", front_image_c_final)
            # cv2.imshow("image_f_final", front_image_f_final)

            result, score = compare_binary_images(front_image_c_final, front_image_f_final, similarity_threshold)
            error_name = compare3_images(front_image_c_final, front_comp1_final, front_comp2_final, front_comp3_final)
            # print(error_name)
            result2, score2, score_scaled2 = compare_contours(front_c_contours, front_f_contours, similarity_threshold)
            # print(score2)
            # print(score_scaled2)

            final_score = round(score * weight1 + score_scaled2 * weight2, 2)
            # print(str(similarity_threshold <= final_score) + "-> Final Score: " + str(final_score), "type =" + type)
            return final_score ,error_name ,front_image_f_final ,front_image_c_final





def compare3_images(test_image,comp1,comp2,comp3,similarity_threshold=0.9):
    result1, score1 = compare_binary_images(test_image, comp1, similarity_threshold)
    result2, score2 = compare_binary_images(test_image, comp2, similarity_threshold)
    result3, score3 = compare_binary_images(test_image, comp3, similarity_threshold)
    dst = [[1,score1],[2,score2],[3,score3]]
    dst.sort(key=operator.itemgetter(1))
    # print(dst)
    if dst[2][0] == 1:
        return "1"
    elif dst[2][0] == 2:
        return "4"
    else:
        return "3"


def error_compare(type, error_name,image_c_final,image_f_final):
    cv2.imshow("Data Image",image_c_final)
    cv2.imshow("Input Image to be compared",image_f_final)
    cv2.waitKey(0)
    if error_name == "1":
        cv2.imwrite("examples/support_error.jpg", image_f_final)
        error_found = cv2.imread("examples/support_error.jpg")
        unSupport(error_found)
        print("Error name: Support is missing")
        cv2.waitKey(0)

    elif error_name == "3":
        print(image_c_final.shape)
        print(image_f_final.shape)

        cv2.imwrite("examples/retrection_error.jpg", image_f_final)
        cv2.imwrite("examples/retrection_true.jpg", image_c_final)
        error_image = cv2.imread("examples/retrection_error.jpg")
        true_image = cv2.imread("examples/retrection_true.jpg")
        final_im = retrection(true_image, error_image)
        cv2.imshow("Retrection Error", final_im)
        print("Error name: Retrection", final_im)


        cv2.waitKey(0)

    elif error_name == "4":

        image_lift = cv2.imread(r"input_images/4/4_f.jpg")
        img = edgeLift(image_lift)
        cv2.imshow("Edge_lift", img)

        image_normal = center_of_gravity(image_c_final)
        image_lifted = center_of_gravity(image_f_final)
        cv2.imshow("Correct Object Center of gravity", image_normal)
        cv2.imshow("Failrue Object Center of gravity", image_lifted)
       
        print("Error name: Edge of the object is lifted")
        
        cv2.waitKey(0)
