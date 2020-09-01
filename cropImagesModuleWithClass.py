import cv2
from PIL import Image
import os
import shutil

class allVariable:
        main_dir = "yalefaces"
        original_faces_directory = "yalefaces"
        path = os.path.join(main_dir,original_faces_directory )
        filenames = os.listdir(path) ##original filename
        jpgImageDirectory = 'yalefaces/jpgImages'
        croppedImageDirectory= 'yalefaces/croppedImages'
        boxPlotImageDirectory= 'yalefaces/boxPlotOfTheDetectedFace'
        directories=['croppedImages','boxPlotOfTheDetectedFace','jpgImages']

class CropFunctions(allVariable):
    def __init__(self):
        self.main_dir = allVariable.main_dir
        self.original_faces_directory = allVariable.original_faces_directory
        self.path = allVariable.path
        self.filenames = allVariable.filenames
        self.jpgImageDirectory = allVariable.jpgImageDirectory
        self.croppedImageDirectory = allVariable.croppedImageDirectory
        self.boxPlotImageDirectory = allVariable.boxPlotImageDirectory
        self.directories = allVariable.directories
    print("hi")
    ##creating necessary directories
    def createDirectories(self):
        for dir in self.directories:
            if(os.path.exists(os.path.join(self.main_dir,dir))):
                shutil.rmtree(os.path.join(self.main_dir,dir))
                print("[INFO] removed directory %s successfully!"%dir)
                os.mkdir(os.path.join(self.main_dir,dir))
                print("[INFO] created directory %s successfully!"%dir)
            else:
                os.mkdir(os.path.join(self.main_dir,dir))
                print("[INFO] created directory %s successfully!"%dir)

    ## the original images do not have gif extension so it is not openend hence provided gif extension
    def imageToGIF(self, filename):
        os.rename(os.path.join(self.path, filename), os.path.join(self.path, filename + '.gif'))


    ###replace gif with jpg name ## this will only change the name but not format ## for creating name it is created
    def saveToJPGFormat(self,filename):
        last_char_index = filename.rfind("gif")
        new_string = filename[:last_char_index]+"jpg"
        return new_string

    ## convert iamges to jpg format and save in the folder
    def convertImageToJPG(self,filename):
        im = Image.open(os.path.join(self.path,filename))
        # im.save(saveToJPGFormat(filename))
        im.convert('RGB').save(os.path.join('yalefaces/jpgImages/',self.saveToJPGFormat(filename)))
    ##read the images with cv2
    def readImageObjectWithCv2(self,directory,filename):
        # imgg = Image.open(os.path.join('yalefaces/jpgImages/',filename)) ##using PIL
        image = cv2.imread(os.path.join(directory, filename))
        # print("Input Images are:", image)
        return image

    ##read the images with PIL
    def readImageObjectWithPIL(self,directory,filename):
        image = Image.open(os.path.join(directory,filename)) ##using PIL
        return image

    ##convert image to grayscale object
    def convertToGrayScale (self,filename):
        image = cv2.imread(os.path.join(self.jpgImageDirectory, filename))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("grayscale inputs are:", gray)
        return gray

    def boxplotOfTheDetectedFace(self,filename):
        image = cv2.imread(os.path.join(self.jpgImageDirectory, filename))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
        )
        if len(faces) == 0:
            print("No Faces Found")
            return
        else:
            print("[INFO] Found {0} Faces!".format(len(faces)))

            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                status = cv2.imwrite(os.path.join(self.boxPlotImageDirectory, filename), image) ## it will simply draw rectangle on detected
                # without cropping
                print("[INFO] Image faces_detected.jpg written to filesystem: ", status)

    def cropFace(self,filename):
        image = cv2.imread(os.path.join(self.jpgImageDirectory, filename))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
        )
        if len(faces) == 0:
           print("No Faces Found")
           return
        else:
            print("[INFO] Found {0} Faces!".format(len(faces)))

            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi_color = image[y:y + h, x:x + w]
                print("[INFO] Object found. Saving locally.")
                status = cv2.imwrite(os.path.join(self.croppedImageDirectory, filename), roi_color)
                print("[INFO] Image faces_detected.jpg written to filesystem: ", status)
    ##crop all images and save
    def createAllCroppedFaces(self,jpgImageDirectory):
        jpgfilenames = os.listdir(jpgImageDirectory)
        for filename in jpgfilenames:
            self.cropFace(filename)

    def createReactangularBoxForAllImages(self,jpgImageDirectory ):
        jpgfilenames = os.listdir(jpgImageDirectory)
        for filename in jpgfilenames:
            self.boxplotOfTheDetectedFace(filename)

    def createAndSaveAllImagesToJPGFormat(self,filenames):
        print(filenames)
        for filename in filenames:
            self.convertImageToJPG(filename)


if __name__ == '__main__':

    allVf = allVariable()
    cpf = CropFunctions()
    cpf.createDirectories()
    cpf.createAndSaveAllImagesToJPGFormat(allVf.filenames)
    cpf.createAllCroppedFaces(allVf.jpgImageDirectory)
    cpf.createReactangularBoxForAllImages(allVf.jpgImageDirectory)


