import cv2
import filters
import helper
import noise_reducer
import sys


def editor_dashboard(file_name):
    image = cv2.imread('train/' + file_name, cv2.IMREAD_COLOR)
    print(f'''
        Type an integer to apply an operation:
            1. Add a glass
            2. Add a thug glass
            3. Add a AR Nose
            4. Apply Dog like face
            5. Add a mustache
            6. Add a hat
            7. More bright photo
            8. Error Currection
            9. Back
            ''')
    while True:
        option = input('Your choice:')
        if option == '1':
            print('Adding a glass on the eyes...')
            output = filters.glass_filter(image, 'glasses.png')
            helper.show('Output Image', output)
            return output

        elif option == '2':
            print('Adding a thug glass on the eyes...')
            output = filters.glass_filter(image, 'thug.png')
            helper.show('Output Image', output)
            return output

        elif option == '3':
            print('Adding an AR nose...')
            output = filters.nose_filter(image)
            helper.show('Output Image', output)
            return output

        elif option == '4':
            print('Applying a dog like face...')
            output = filters.dog_filter(image)
            helper.show('Output Image', output)
            return output

        elif option == '5':
            print('Adding a mustache...')
            output = filters.mustache_filter(image)
            helper.show('Output Image', output)
            return output

        elif option == '6':
            print('Adding a hat...')
            output = filters.hat_filter(image)
            helper.show('Output Image', output)
            return output

        elif option == '7':
            print('Making your photo brighter...')
            output = filters.equalization_filter(image)
            helper.show('Output Image', output)
            return output

        elif option == '8':
            sigma = float(input('sigma:'))
            kernel_sz = int(input('kernel size:'))
            print('Error correcting in your image...')
            output = noise_reducer.reduce(image, sigma, kernel_sz)
            helper.show('Output Image', output)
            return output

        elif option == '9':
           return None
        else:
            print('Invalid option selected.')

file_name = None
output_image = None

def main_dashboard():
    global file_name
    global output_image
    print(f'''
    Welcome to snapchat AR Photo Editor!
    1. To edit an image
    2. Save the edited image
    3. Open Editing dashboard
    4. exit
    ''')
    option = input('Your choice:')
    if option == '1':
        file_name = input('File name:')
        output_image = editor_dashboard(file_name)
    elif option == '2':
        if output_image is not None:
            output_file = input('Output file name:')
            cv2.imwrite('train/' + output_file, output_image)
            print('Saved successfully!')
        else:
            print('No operation is applied in the image!')
    elif option == '3':
        output_image = editor_dashboard(file_name)
    elif option == '4':
        sys.exit(0)
    else:
        print('Invalid option selected.')


if __name__ == '__main__':
    while True:
        main_dashboard()
