import cv2
import numpy as np
import matplotlib.pyplot as plot
from PIL import Image

tryb = int(input("Wybierz tryb działania ( 0 - dokodowanie pliku , 1 - anonimizacja pliku, 2 - FFT) : "))


def dekodowanie(wejscie, wyjscie):
    file = open(wejscie, "rb")
    plik = open(wyjscie,'w')

    print("::PNG SIGNATURE::")

    for i in range(8):
        signature = file.read(1)
        print(signature.hex())

    chunk_length = int.from_bytes(file.read(4), byteorder='big')
    chunk_type = bytearray.fromhex(file.read(4).hex()).decode()
    chunk_data = file.read(chunk_length)
    crc = file.read(4)

    data_for_FFT = ""
    filter_method = 0;

    while chunk_data:
        plik.write("\n\n::NEXT CHUNK::")
        plik.write("\nChunk length: ")
        plik.write(str(chunk_length))
        plik.write("\nChunk type: ")
        plik.write(str(chunk_type))

        print("\n::NEXT CHUNK::")
        print("Chunk length: ", chunk_length)
        print("Chunk type: ", chunk_type)

        if chunk_type.upper() in {"IHDR","IDAT","PLTE","PHYS","TEXT","CHRM"}:
            if chunk_type.upper() == "IHDR":
                variable = int(chunk_data.hex()[0:8], 16)
                plik.write("\nImage Width: "+str(variable))
                print("Image Width: ", variable)
                variable = int(chunk_data.hex()[8:16], 16)
                plik.write("\nImage Height: "+ str(variable))
                print("Image Height: ", variable)
                variable = int(chunk_data.hex()[16:18], 16)
                plik.write("\nBit Depht: "+str(variable))
                print("Bit Depht: ", variable)
                variable = int(chunk_data.hex()[18:20], 16)
                plik.write("\nColor Type: "+ str(variable))
                print("Color Type: ", variable)
                variable = int(chunk_data.hex()[20:22], 16)
                plik.write("\nCompression Method: "+ str(variable))
                print("Compression Method: ", variable)
                variable = int(chunk_data.hex()[22:24], 16)
                plik.write("\nFilter Method: "+ str(variable))
                print("Filter Method: ", variable)
                filter_method = int(chunk_data.hex()[24:26], 16)
                plik.write("\nInterlace Method: "+ str(filter_method))
                print("Interlace Method: ", filter_method)

            if chunk_type.upper() == "IDAT":
                variable = chunk_data.hex()
                print("Chunk IDAT data: IDAT DATA COVERS OTHER INFO", )
                plik.write("\nChunk IDAT data: IDAT DATA COVERS OTHER INFO")
                data_for_FFT = data_for_FFT+variable

            if chunk_type.upper() == "PLTE":
                print("Palette:", chunk_data.hex())
                plik.write("\nPalette:"+ str(chunk_data.hex()))

            if chunk_type.upper() == "PHYS":
                variable = int(chunk_data.hex()[0:8], 16)
                print("PixelsPerUnitX: ", variable)
                plik.write("\nPixelsPerUnitX: "+ str(variable))
                variable = int(chunk_data.hex()[8:16], 16)
                print("PixelsPerUnitY: ", variable)
                plik.write("\nPixelsPerUnitY: "+ str(variable))
                variable = int(chunk_data.hex()[16:18], 16)
                print("PixelUnits: ", variable, "   ( if equal 1 it's meters, if equal 0 it's unknown )")
                plik.write("\nPixelUnits: "+ str(variable)+ "\t( if equal 1 it's meters, if equal 0 it's unknown )")

            if chunk_type.upper() in {"TEXT","ITXT","ZTXT"}:
                print("Chunk data: ", bytearray.fromhex(chunk_data.hex()).decode())
                plik.write("\nChunk data: "+ bytearray.fromhex(chunk_data.hex()).decode())

            if chunk_type.upper() == "CHRM":
                variable = int(chunk_data.hex()[0:4], 16)
                print("White Point x:",variable)
                plik.write("\nWhite Point x:"+str(variable))
                variable = int(chunk_data.hex()[4:8], 16)
                print("White Point y:", variable)
                plik.write("\nWhite Point y:"+str(variable))
                variable = int(chunk_data.hex()[8:12], 16)
                print("Red x:", variable)
                plik.write("\nRed x:"+ str(variable))
                variable = int(chunk_data.hex()[12:16], 16)
                print("Red y:", variable)
                plik.write("\nRed y:"+ str(variable))
                variable = int(chunk_data.hex()[16:20], 16)
                print("Green x:", variable)
                plik.write("\nGreen x:"+str(variable))
                variable = int(chunk_data.hex()[20:24], 16)
                print("Green y:", variable)
                plik.write("\nGreen y:"+ str(variable))
                variable = int(chunk_data.hex()[24:28], 16)
                print("Blue x:", variable)
                plik.write("\nBlue x:"+str(variable))
                variable = int(chunk_data.hex()[28:32], 16)
                print("Blue y:", variable)
                plik.write("\bBlue y:"+str(variable))

        else:
           print(chunk_data.hex())
           plik.write("\n"+str(chunk_data.hex()))

        print("CRC: ", crc.hex())
        plik.write("\nCRC: "+ crc.hex())

        chunk_length = int.from_bytes(file.read(4), byteorder='big')
        chunk_type = bytearray.fromhex(file.read(4).hex()).decode()
        chunk_data = file.read(chunk_length)
        crc = file.read(4)

    print("\n::NEXT CHUNK::")
    plik.write("\n\n::NEXT CHUNK::")
    print("Chunk length: ", chunk_length)
    plik.write("\nChunk length: "+str(chunk_length))
    print("Chunk type: ", chunk_type)
    plik.write("\nChunk type: "+ str(chunk_type))
    print("CRC: ", crc.hex())
    plik.write("\nCRC: "+crc.hex())

    print("\n\nData for FFT:", data_for_FFT)
    # Tutaj próbuję zapisać do pliku tylko te chunki główne, ale niezbyt wiem jak to zapisać/ w jakim formacie. 
    # Nie wiem czy nie trzeba ich jakoś kodować i czy nie łatwiej będzie wyrzucić zbędne informacje z głównego pliku niż tworzyć nowy


def anonimizacja(wejscie, wyjscie):
    chunks = input("Podaj nazwe pliku wyjsciowego, gdzie zapisane zostana zdekodowane chunki przed anonimizacja: ")
    dekodowanie(wejscie, chunks)
    old_png = open(wejscie, "rb")
    new_png = open(wyjscie,'wb')
    for i in range(8):
         signature = old_png.read(1)
         new_png.write(signature)
    chunk_length = old_png.read(4)
    chunk_length2 = int.from_bytes(chunk_length, byteorder='big')
    chunk_type = old_png.read(4)
    chunk_type2 = bytearray.fromhex(chunk_type.hex()).decode()
    chunk_data = old_png.read(chunk_length2)
    crc = old_png.read(4)
    while chunk_data:
        if chunk_type2.upper() in {"IHDR","IDAT","PLTE"}:
            new_png.write(chunk_length)
            new_png.write(chunk_type)
            new_png.write(chunk_data)
            new_png.write(crc)

        chunk_length = old_png.read(4)
        chunk_length2 = int.from_bytes(chunk_length, byteorder='big')
        chunk_type = old_png.read(4)
        chunk_type2 = bytearray.fromhex(chunk_type.hex()).decode()
        chunk_data = old_png.read(chunk_length2)
        crc = old_png.read(4)
    new_png.write(chunk_length)
    new_png.write(chunk_type)
    new_png.write(chunk_data)
    new_png.write(crc)
    new_png.close()
   
    chunks = input("Podaj nazwe pliku wyjsciowego, gdzie zapisane zostana zdekodowane chunki po anonimizacji: ")
    dekodowanie(wyjscie, chunks)

def showImage(file):
    # Funkcje z biblioteki PILLOW wyświetlają oryginalny obraz
    #read the image
    im = Image.open(file)
    #show image
    im.show()


# Funkcje z bibliotek/modułów cv2 numpy i matplot... wyświetlają widmo ampl. i fazowe
def doFourierTransform(file):
    plot.figure(figsize=(6.4*5, 4.8*5))

    image1 = cv2.imread(file, 0)
    image2 = np.fft.fft2(image1)
    image3 = np.fft.fftshift(image2)
    image4 = np.angle(image3)
    plot.subplot(142), plot.imshow(np.log(1+np.abs(image3)), "gray"), plot.title("widmo amplitudowe")
    plot.subplot(143), plot.imshow(np.log(1+np.abs(image4)), "gray"), plot.title("widmo fazowe")
    plot.show()

if tryb==0:
    input_file = input("Podaj nazwe pliku wejsciowego PNG: ")
    output_file = input("Podaj nazwe pliku wyjsciowego, gdzie zapisane zostana zdekodowane chunki: ")
    dekodowanie(input_file, output_file)
if tryb ==1:
    input_file = input("Podaj nazwe pliku wejsciowego PNG: ")
    showImage(input_file)
    wyjscie = input('Podaj nazwe pliku wyjsciowego png po anonimizacji: ')
    anonimizacja(input_file, wyjscie)
    showImage(wyjscie)
if tryb ==2:
    transform_file = input("Podaj nazwe pliku którego transformacji należy dokonać: ")
    doFourierTransform(transform_file)









