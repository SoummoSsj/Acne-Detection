import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Load the cascades
face_cascade = cv2.CascadeClassifier('frontal_face.xml')
eye_cascade = cv2.CascadeClassifier('frontal_eye.xml')
mouth_cascade = cv2.CascadeClassifier('mouth.xml')

def detect_face(image, reduction_percentage=0.1):
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10, minSize=(80, 80))
    reduced_faces = []
    
    for (x, y, w, h) in faces:
        reduction_w = int(w * reduction_percentage)
        reduction_h = int(h * reduction_percentage)
        
        new_x = x + reduction_w // 2
        new_y = y + reduction_h // 2
        new_w = w - reduction_w
        new_h = h - reduction_h
        
        reduced_faces.append((new_x, new_y, new_w, new_h))
    
    return reduced_faces

def detect_eye(image):
    eyes = eye_cascade.detectMultiScale(image, scaleFactor=1.05, minNeighbors=10, minSize=(30, 30))
    return eyes

def detect_mouth(image):
    mouths = mouth_cascade.detectMultiScale(image, scaleFactor=1.5, minNeighbors=11, minSize=(30, 30))
    return mouths

def custom_threshold(image, threshold_value):
    thresholded = np.zeros_like(image)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y, x] > threshold_value:
                thresholded[y, x] = 255
            else:
                thresholded[y, x] = 0
    return thresholded

def custom_dilate(image, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    pad = kernel_size // 2
    dilated = np.zeros_like(image)
    
    for y in range(pad, image.shape[0] - pad):
        for x in range(pad, image.shape[1] - pad):
            if np.any(image[y-pad:y+pad+1, x-pad:x+pad+1] == 255):
                dilated[y, x] = 255
                
    return dilated

def detect_acne(face_region, threshold_value=20.0):
    gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    blurred_face = cv2.GaussianBlur(gray_face, (15, 15), 0.3)
    acne_mask = cv2.absdiff(gray_face, blurred_face)
    acne_mask = custom_threshold(acne_mask, threshold_value)
    contours, _ = cv2.findContours(acne_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_acne_mask = np.zeros_like(acne_mask)
    for contour in contours:
        if cv2.contourArea(contour) < 5:
            cv2.drawContours(filtered_acne_mask, [contour], -1, 255, -1)
    filtered_acne_mask = custom_dilate(filtered_acne_mask, kernel_size=3)
    return filtered_acne_mask

def inpaint_telea_manual(src, mask, radius):
    inpainted_img = src.copy()
    mask = np.uint8(mask)  # Ensure mask is of type uint8
    h, w = src.shape[:2]

    # Normalize mask to binary (0 and 1)
    mask = mask / 255

    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0:
                sum_num = 0
                sum_den = 0

                for j in range(-radius, radius + 1):
                    for i in range(-radius, radius + 1):
                        if x + i >= 0 and x + i < w and y + j >= 0 and y + j < h:
                            if mask[y + j, x + i] == 1:
                                dist = np.sqrt(i * i + j * j)
                                weight = np.exp(-dist / radius)

                                num = weight * inpainted_img[y + j, x + i]
                                den = weight

                                sum_num += num
                                sum_den += den

                if sum_den > 0:
                    inpainted_img[y, x] = sum_num / sum_den

    return inpainted_img

def process_image(image, inpainting_method, inpainting_radius=3, threshold_value=20.0):
    original_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detect_face(image)
    if len(faces) == 0:
        return None, None, "No faces detected"

    for i, (x, y, w, h) in enumerate(faces):
        face_region = image[y:y+h, x:x+w]
        gray_face_region = gray[y:y+h, x:x+w]

        eyes = detect_eye(face_region)
        eye_regions = []
        eye_margin = 0.1  # 10% margin around eyes
        for j, (ex, ey, ew, eh) in enumerate(eyes):
            ex_margin = int(ew * eye_margin)
            ey_margin = int(eh * eye_margin)
            ex_new = max(0, ex - ex_margin)
            ey_new = max(0, ey - ey_margin)
            ew_new = min(face_region.shape[1] - ex_new, ew + 2 * ex_margin)
            eh_new = min(face_region.shape[0] - ey_new, eh + 2 * ey_margin)

            eye_regions.append((ex_new, ey_new, ew_new, eh_new, face_region[ey_new:ey_new+eh_new, ex_new:ex_new+ew_new].copy()))

        mouths = detect_mouth(face_region)
        mouth_regions = []
        for k, (mx, my, mw, mh) in enumerate(mouths):
            mouth_regions.append((mx, my, mw, mh, face_region[my:my+mh, mx:mx+mw].copy()))

        mask = np.ones(face_region.shape[:2], dtype=np.uint8) * 255
        for (ex_new, ey_new, ew_new, eh_new, _) in eye_regions:
            mask[ey_new:ey_new+eh_new, ex_new:ex_new+ew_new] = 0
        for (mx, my, mw, mh, _) in mouth_regions:
            mask[my:my+mh, mx:mx+mw] = 0

        acne_mask = detect_acne(face_region, threshold_value=threshold_value)

        combined_mask = cv2.bitwise_and(acne_mask, acne_mask, mask=mask)

        if inpainting_method == "opencv":
            inpainted_face = cv2.inpaint(face_region, combined_mask, inpainting_radius, cv2.INPAINT_TELEA)
        elif inpainting_method == "manual":
            inpainted_face = inpaint_telea_manual(face_region, combined_mask, inpainting_radius)
        
        for (ex_new, ey_new, ew_new, eh_new, eye_region) in eye_regions:
            inpainted_face[ey_new:ey_new+eh_new, ex_new:ex_new+ew_new] = eye_region
        for (mx, my, mw, mh, mouth_region) in mouth_regions:
            inpainted_face[my:my+mh, mx:mx+mw] = mouth_region

        image[y:y+h, x:x+w] = inpainted_face

    return original_image, image, None

def clone_region(source_region, target_region):
    # Simple copy-paste for demonstration
    target_region[:] = source_region[:]

def load_image(image_file):
    img = Image.open(image_file)
    return np.array(img)

st.set_page_config(page_title="Acne Detection and Inpainting", page_icon="🖼")

st.title("Acne Detection and Inpainting")
st.write("""
This app detects acne on your face and performs inpainting to remove it. Upload an image to see the magic!
""")

st.sidebar.header("Upload Image")
input_image_file = st.sidebar.file_uploader("Upload Input Image", type=["png", "jpg", "jpeg"])

inpainting_radius = st.sidebar.slider("Inpainting Radius", min_value=1, max_value=30, value=3, step=1)
threshold_value = st.sidebar.slider("Threshold Value", min_value=0.0, max_value=255.0, value=20.0, step=0.1)

st.sidebar.header("Inpainting Method")
inpainting_method = st.sidebar.radio(
    "Choose the inpainting method",
    ("None", "OpenCV (Faster)", "Manual (Slower)")
)

if input_image_file:
    input_image = load_image(input_image_file)

    st.subheader("Uploaded Image")
    st.image(input_image, caption='Input Image', use_column_width=True)

    input_image_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    
    if inpainting_method != "None":
        if inpainting_method == "OpenCV (Faster)":
            method = "opencv"
        else:
            method = "manual"
            
        original_image, processed_image, error_message = process_image(input_image_bgr, method, inpainting_radius, threshold_value)
        
        if error_message:
            st.error(error_message)
        else:
            st.subheader("Processing Steps")
            
            st.image(original_image, caption='Original Image', use_column_width=True)

            faces = detect_face(input_image_bgr)
            for i, (x, y, w, h) in enumerate(faces):
                face_region = input_image_bgr[y:y+h, x:x+w]
                gray_face_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                
                # Display detected face region
                st.image(cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB), caption=f'Face Region {i}', use_column_width=True)
                
                # Display grayscale eye regions without boxes
                for (ex, ey, ew, eh) in detect_eye(face_region):
                    eye_region = face_region[ey:ey+eh, ex:ex+ew]
                    st.image(cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB), caption='Eye Region', use_column_width=True)

                # Display grayscale mouth regions without boxes
                for (mx, my, mw, mh) in detect_mouth(face_region):
                    mouth_region = face_region[my:my+mh, mx:mx+mw]
                    st.image(cv2.cvtColor(mouth_region, cv2.COLOR_BGR2RGB), caption='Mouth Region', use_column_width=True)

                # Display acne mask
                acne_mask = detect_acne(face_region, threshold_value=threshold_value)
                st.image(acne_mask, caption='Acne Mask', use_column_width=True)

                # Display combined mask
                mask = np.ones(face_region.shape[:2], dtype=np.uint8) * 255
                for (ex, ey, ew, eh) in detect_eye(face_region):
                    mask[ey:ey+eh, ex:ex+ew] = 0
                for (mx, my, mw, mh) in detect_mouth(face_region):
                    mask[my:my+mh, mx:mx+mw] = 0
                
                st.image(mask, caption='Face Mask', use_column_width=True)

                combined_mask = cv2.bitwise_and(acne_mask, acne_mask, mask=mask)
                st.image(combined_mask, caption='Combined Mask', use_column_width=True)

                if inpainting_method == "OpenCV (Faster)":
                    inpainted_face = cv2.inpaint(face_region, combined_mask, inpainting_radius, cv2.INPAINT_TELEA)
                else:
                    inpainted_face = inpaint_telea_manual(face_region, combined_mask, inpainting_radius)
                
                st.image(cv2.cvtColor(inpainted_face, cv2.COLOR_BGR2RGB), caption='Inpainted Face', use_column_width=True)
            
            st.subheader("Processed Image")
            st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption='Processed Image', use_column_width=True)

            result_image_pil = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
            buf = io.BytesIO()
            result_image_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()

            # Display side-by-side comparison
            st.subheader("Comparison: Original vs Processed Image")
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), caption='Original Image', use_column_width=True)
            with col2:
                st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption='Processed Image', use_column_width=True)

            st.download_button(
                label="Download Processed Image",
                data=byte_im,
                file_name="processed_image.png",
                mime="image/png"
            )
else:
    st.sidebar.info("Please upload an input image to proceed.")
