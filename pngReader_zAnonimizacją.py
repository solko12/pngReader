import math
import random
import time
import zlib

import cv2
import numpy as np
import matplotlib.pyplot as plot
from PIL import Image


keys={"public": (0,0), "private": (0,0)}


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

            if chunk_type.upper() == "PLTE":
                for i in range(0,chunk_length,3):
                    if i == 0:
                        print("\nPalette nb. 1")
                    else:
                        print("Palette nb. "+str(int(i/3+1)))
                    print("Red: "+str(int.from_bytes(chunk_data[i:i+1], byteorder='big')))
                    print("Green: "+str(int.from_bytes(chunk_data[i+1:i+2], byteorder='big')))
                    print("Blue: "+str(int.from_bytes(chunk_data[i+2:i+3], byteorder='big')))
                    if i == 0:
                        plik.write("\nPalette nb. 1")
                    else:
                        plik.write("Palette nb. "+str(int(i/3+1)))
                    plik.write("\nRed: "+str(int.from_bytes(chunk_data[i:i+1], byteorder='big')))
                    plik.write("\nGreen: "+str(int.from_bytes(chunk_data[i+1:i+2], byteorder='big')))
                    plik.write("\nBlue: "+str(int.from_bytes(chunk_data[i+2:i+3], byteorder='big'))+"\n")

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
        plik.write("\nCRC: " + crc.hex())

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
    print("Chunk data: ", chunk_data)
    plik.write("\nChunkt data:"+str(chunk_data))
    print("CRC: ", crc.hex())
    plik.write("\nCRC: "+crc.hex())
    plik.close()


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
    old_png.close()
   
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
    plot.subplot(121), plot.imshow(np.log(1+np.abs(image3)), "gray"), plot.title("widmo amplitudowe")
    plot.axis('off')
    plot.subplot(122), plot.imshow(np.log(1+np.abs(image4)), "gray"), plot.title("widmo fazowe")
    plot.axis('off')
    plot.show()

def findPrime(primeBits):
    try:
        key = random.getrandbits(primeBits)
        if(isPrime(key)==True):
            return key
        else:
            while(isPrime(key)==False):
                key = random.getrandbits(primeBits)
    except OverflowError:
        ans = float('inf')
    return key


def nwd(a, b):
    while b:
        a, b = b, a%b
    return a


def isPrime(number):
    flag = False
    if number==1:
        return False
    #Preselection checking if it's even number
    if number % 2 == 0:
        return False
    else:
        #Preselection for max 10000 odd numbers
        if(math.floor(math.sqrt(number))<10000):
            end=math.floor(math.sqrt(number))
        else:
            end=10000
        for i in range(3, end, 2):
            if number % i == 0:
                return False
        #When number is primary in previous range check, check it by using Miller-Rabin Test
        #Find max power of 2 and maximum multiply in number-1
        s=0
        d=number-1
        while(d%2==0):
            s=s+1
            d=d/2
        flag=True
        # Miller-Rabin Test n-times
        # All of the single tests reduces error possibility by multiply it by 1/4
        # So error possibility is compute by (1/4)^n
        n = 100
        for i in range(1, n, 1):
            a = random.randrange(2, number - 2)
            x = math.pow(a, d) % number
            if(x==1 or x==number-1):
                while x == 1 or x == number-1:
                    a = random.randrange(2, number - 2)
                    x = math.pow(a, d) % number
            j=1
            while j<s and x!=number-1:
                x=math.pow(x,2)%number
                if x==1:
                    return False
                j=j+1
            if x!=number-1:
                return False
        #I forgot name of these test
        for i in range(1,10,1):
            a = random.randrange(2,number-1)
            if nwd(number,a)!=1:
                return False
            if math.pow(a,number-1)%number!=1:
                return False
        #If number pass all tests return true
        return True


def extendedEuklides(a, b):
    u = 1
    w = a
    x = 0
    z = b
    while w != 0:
        if w < z:
            u, x = x, u
            w, z = z, w
        q = w//z
        u = u - q*x
        w = w - q*z
    if z != 1:
        return False
    if x < 0:
        x = x+b
    return x


def generateRSA():
    p = findPrime(32)
    q = findPrime(32)
    fi = (p-1)*(q-1)
    n = p*q
    e = random.randrange(1, n)
    while nwd(e, fi) != 1:
        e = random.randrange(1, n)
    d = extendedEuklides(fi, e)
    return {"public": (e, n), "private": (d, n)}


def encodeNumber(numberToEncode):
    return (numberToEncode ^ keys["public"][0]) % keys["public"][1]

def encodePicture(wejscie, wyjscie):
    old_png = open(wejscie, "rb")
    new_png = open(wyjscie, 'wb')
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
        if chunk_type2.upper() == "IDAT":
            new_png.write(chunk_length)
            new_png.write(chunk_type)
            i = 1
            chunk_encoded_data = ""
            for singleNumber in chunk_data:
                # if i == 1:
                #     print("R: " + str(singleNumber))
                # elif i == 2:
                #     print("G: " + str(singleNumber))
                # elif i == 3:
                #     print("B: " + str(singleNumber))
                # elif i == 4:
                #     print("A: " + str(singleNumber))
                #     i = 0
                # chunk_encoded_data += str(encodeNumber(singleNumber))
                i += 1
            new_png.write(bytes(chunk_encoded_data.encode()))

        else:
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
    old_png.close()

filein = "PNGFile.png"
fileout = "out.png"

start = time.time()
keys = generateRSA()
end = time.time()
print("Public key: " + str(keys["public"]) + "\nPrivate key: " + str(keys["private"]))
print("Finding key time: " + str(end-start))

encodePicture(filein, fileout)

tryb = int(input("Wybierz tryb działania "
                 "\nDostepne opcje: \n0 - dokodowanie pliku, \n1 - anonimizacja pliku, \n2 - FFT, "
                 "\n3 - wyswietl zdjecie,\n4 - koduj obraz \n9 - wyjscie \nWybor: "))


while(tryb!=9):
    if tryb==0:
        input_file = input("Podaj nazwe pliku wejsciowego PNG: ")
        output_file = input("Podaj nazwe pliku wyjsciowego, gdzie zapisane zostana zdekodowane chunki: ")
        dekodowanie(input_file, output_file)
    elif tryb ==1:
        input_file = input("Podaj nazwe pliku wejsciowego PNG: ")
        wyjscie = input('Podaj nazwe pliku wyjsciowego png po anonimizacji: ')
        anonimizacja(input_file, wyjscie)
    elif tryb ==2:
        transform_file = input("Podaj nazwe pliku którego transformacji należy dokonać: ")
        doFourierTransform(transform_file)
    elif tryb == 3:
        picture_to_show = input("Podaj nazwe pliku png do wyswietlenia: ")
        showImage(picture_to_show)
    elif tryb == 4:
        picture_to_encode = input("Podaj nazwe pliku do zakodowania: ")
        picture_after_encode = input("Podaj nazwe pliku wynikowego: ")
        encodePicture(picture_to_encode, picture_after_encode)
    else:
        print("Błędny tryb! Wybierz z listy prawidłowy!!")
    tryb = int(input("Wybierz tryb działania "
                     "\nDostepne opcje: \n0 - dokodowanie pliku, \n1 - anonimizacja pliku, \n2 - FFT, "
                     "\n3 - wyswietl zdjecie, \n9 - wyjscie \nWybor: "))









