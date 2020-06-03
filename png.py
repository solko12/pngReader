import random
import struct
import time
import cv2
import numpy as np
import matplotlib.pyplot as plot
import rsa as rsa
from PIL import Image, ImageDraw, ImageFile, ImagePalette, _binary
import zlib
import io

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import binascii

from rsa import PublicKey, PrivateKey

lastChunkSize = 0
primeBitsCount = 512
keys = {"public": (0, 0), "private": (0, 0, 0, 0)}

def dekodowanie(wejscie, wyjscie):
    file = open(wejscie, "rb")
    plik = open(wyjscie, 'w')

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

        if chunk_type.upper() in {"IHDR", "IDAT", "PLTE", "PHYS", "TEXT", "CHRM"}:
            if chunk_type.upper() == "IHDR":
                variable = int(chunk_data.hex()[0:8], 16)
                plik.write("\nImage Width: " + str(variable))
                print("Image Width: ", variable)
                variable = int(chunk_data.hex()[8:16], 16)
                plik.write("\nImage Height: " + str(variable))
                print("Image Height: ", variable)
                variable = int(chunk_data.hex()[16:18], 16)
                plik.write("\nBit Depht: " + str(variable))
                print("Bit Depht: ", variable)
                variable = int(chunk_data.hex()[18:20], 16)
                plik.write("\nColor Type: " + str(variable))
                print("Color Type: ", variable)
                variable = int(chunk_data.hex()[20:22], 16)
                plik.write("\nCompression Method: " + str(variable))
                print("Compression Method: ", variable)
                variable = int(chunk_data.hex()[22:24], 16)
                plik.write("\nFilter Method: " + str(variable))
                print("Filter Method: ", variable)
                filter_method = int(chunk_data.hex()[24:26], 16)
                plik.write("\nInterlace Method: " + str(filter_method))
                print("Interlace Method: ", filter_method)

            if chunk_type.upper() == "IDAT":
                variable = chunk_data.hex()
                print("Chunk IDAT data: IDAT DATA COVERS OTHER INFO", )
                plik.write("\nChunk IDAT data: " + str(variable))

            if chunk_type.upper() == "PLTE":
                for i in range(0, chunk_length, 3):
                    if i == 0:
                        print("\nPalette nb. 1")
                    else:
                        print("Palette nb. " + str(int(i / 3 + 1)))
                    print("Red: " + str(int.from_bytes(chunk_data[i:i + 1], byteorder='big')))
                    print("Green: " + str(int.from_bytes(chunk_data[i + 1:i + 2], byteorder='big')))
                    print("Blue: " + str(int.from_bytes(chunk_data[i + 2:i + 3], byteorder='big')))
                    if i == 0:
                        plik.write("\nPalette nb. 1")
                    else:
                        plik.write("Palette nb. " + str(int(i / 3 + 1)))
                    plik.write("\nRed: " + str(int.from_bytes(chunk_data[i:i + 1], byteorder='big')))
                    plik.write("\nGreen: " + str(int.from_bytes(chunk_data[i + 1:i + 2], byteorder='big')))
                    plik.write("\nBlue: " + str(int.from_bytes(chunk_data[i + 2:i + 3], byteorder='big')) + "\n")

            if chunk_type.upper() == "PHYS":
                variable = int(chunk_data.hex()[0:8], 16)
                print("PixelsPerUnitX: ", variable)
                plik.write("\nPixelsPerUnitX: " + str(variable))
                variable = int(chunk_data.hex()[8:16], 16)
                print("PixelsPerUnitY: ", variable)
                plik.write("\nPixelsPerUnitY: " + str(variable))
                variable = int(chunk_data.hex()[16:18], 16)
                print("PixelUnits: ", variable, "   ( if equal 1 it's meters, if equal 0 it's unknown )")
                plik.write("\nPixelUnits: " + str(variable) + "\t( if equal 1 it's meters, if equal 0 it's unknown )")

            if chunk_type.upper() in {"TEXT", "ITXT", "ZTXT"}:
                print("Chunk data: ", bytearray.fromhex(chunk_data.hex()).decode())
                plik.write("\nChunk data: " + bytearray.fromhex(chunk_data.hex()).decode())

            if chunk_type.upper() == "CHRM":
                variable = int(chunk_data.hex()[0:4], 16)
                print("White Point x:", variable)
                plik.write("\nWhite Point x:" + str(variable))
                variable = int(chunk_data.hex()[4:8], 16)
                print("White Point y:", variable)
                plik.write("\nWhite Point y:" + str(variable))
                variable = int(chunk_data.hex()[8:12], 16)
                print("Red x:", variable)
                plik.write("\nRed x:" + str(variable))
                variable = int(chunk_data.hex()[12:16], 16)
                print("Red y:", variable)
                plik.write("\nRed y:" + str(variable))
                variable = int(chunk_data.hex()[16:20], 16)
                print("Green x:", variable)
                plik.write("\nGreen x:" + str(variable))
                variable = int(chunk_data.hex()[20:24], 16)
                print("Green y:", variable)
                plik.write("\nGreen y:" + str(variable))
                variable = int(chunk_data.hex()[24:28], 16)
                print("Blue x:", variable)
                plik.write("\nBlue x:" + str(variable))
                variable = int(chunk_data.hex()[28:32], 16)
                print("Blue y:", variable)
                plik.write("\bBlue y:" + str(variable))

        else:
            print(chunk_data.hex())
            plik.write("\n" + str(chunk_data.hex()))

        print("CRC: ", crc.hex())
        plik.write("\nCRC: " + crc.hex())

        chunk_length = int.from_bytes(file.read(4), byteorder='big')
        chunk_type = bytearray.fromhex(file.read(4).hex()).decode()
        chunk_data = file.read(chunk_length)
        crc = file.read(4)

    print("\n::NEXT CHUNK::")
    plik.write("\n\n::NEXT CHUNK::")
    print("Chunk length: ", chunk_length)
    plik.write("\nChunk length: " + str(chunk_length))
    print("Chunk type: ", chunk_type)
    plik.write("\nChunk type: " + str(chunk_type))
    print("Chunk data: ", chunk_data)
    plik.write("\nChunkt data:" + str(chunk_data))
    print("CRC: ", crc.hex())
    plik.write("\nCRC: " + crc.hex())
    plik.close()


def anonimizacja(wejscie, wyjscie):
    chunks = input("Podaj nazwe pliku wyjsciowego, gdzie zapisane zostana zdekodowane chunki przed anonimizacja: ")
    dekodowanie(wejscie, chunks)
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
        if chunk_type2.upper() in {"IHDR", "IDAT", "PLTE"}:
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
    # read the image
    im = Image.open(file)
    # show image
    im.show()


# Funkcje z bibliotek/modułów cv2 numpy i matplot... wyświetlają widmo ampl. i fazowe
def doFourierTransform(file):
    plot.figure(figsize=(6.4 * 5, 4.8 * 5))

    image1 = cv2.imread(file, 0)
    image2 = np.fft.fft2(image1)
    image3 = np.fft.fftshift(image2)
    image4 = np.angle(image3)
    plot.subplot(121), plot.imshow(np.log(1 + np.abs(image3)), "gray"), plot.title("widmo amplitudowe")
    plot.axis('off')
    plot.subplot(122), plot.imshow(np.log(1 + np.abs(image4)), "gray"), plot.title("widmo fazowe")
    plot.axis('off')
    plot.show()


# Function searching for prime number in number of bits got in argument primeBits
def findPrime(primeBits):
    try:
        # There is finding key which is checked by isPrime for being prime
        key = random.getrandbits(primeBits)
        if isPrime(key):
            # If is return it
            return key
        else:
            # If not, look for another candidate
            while not isPrime(key):
                key = random.getrandbits(primeBits)
    # It's helpful for working in big numbers
    except OverflowError:
        ans = float('inf')
    return key


# Function is looking for biggest common divide for a and b, it's implementation from web
def nwd(a, b):
    while b:
        a, b = b, a % b
    return a


# Function checked for being prime by number in k Miller-Rabin's tests
def isPrime(number, k=128):
    # Test if n is not even.
    # But care, 2 is prime !
    if number == 2 or number == 3:
        return True
    if number <= 1 or number % 2 == 0:
        return False
    # find d and s
    s = 0
    d = number - 1
    while d & 1 == 0:
        s += 1
        d //= 2
    # do k Miller-Rabin's tests
    for _ in range(k):
        a = random.randrange(2, number - 1)
        x = pow(a, d, number)
        if x != 1 and x != number - 1:
            j = 1
            while j < s and x != number - 1:
                x = pow(x, 2, number)
                if x == 1:
                    return False
                j += 1
            if x != number - 1:
                return False
    return True


# Function is implementation of extended Euklides algorithm
def extendedEuklides(a, b):
    # Start initial values
    u = 1
    w = a
    x = 0
    z = b
    while w != 0:
        # if w<z swap u with x and w with z
        if w < z:
            u, x = x, u
            w, z = z, w
        # make new values of q, u and w
        q = w // z
        u = u - q * x
        w = w - q * z
    # If z!=1 there is not number x
    if z != 1:
        return False
    # if x < 0 make it positive by adding b
    if x < 0:
        x = x + b
    return x


# Function generates RSA keys
def generateRSA():
    # Find two big prime numbers
    p = findPrime(primeBitsCount)
    q = findPrime(primeBitsCount)
    # Compute fi function
    fi = (p - 1) * (q - 1)
    # And n
    n = p * q
    # Look for e<n which is basicly dividable by fi
    e = random.randrange(1, n)
    while nwd(e, fi) != 1:
        e = random.randrange(1, n)
    # Compute d
    d = extendedEuklides(e, fi)
    # Return keys Public(e,n) and Private(d,n)
    return {"public": {"e": e, "n": n}, "private": {"d": d, "n": n, "p":p,"q":q}}


# Function encrypt number got in argument by RSA algorithm
def encryptNumber(number, e, n):
    return pow(number, e, n)


# Function decrypt number got in argument by RSA algorithm
def decryptNumber(number, d, n):
    return pow(number, d, n)


# Function decrypt data got in argument by ECB method
def decryptData(imageData, blockSize, realLength, d, n):
    newIdat = ""
    i = 0
    # Encrypted data is 4 times bigger than not encrypted
    blockSize *= 4
    # While there is still data
    while i < realLength:
        # Get block in correct size
        block = imageData[i:i + blockSize]
        # Change into int
        decBlockInInt = int(block, 16)
        # Decrypt number
        decryptedNumber = decryptNumber(decBlockInInt, d, n)
        # And make it hex
        decryptedBlock = format(decryptedNumber, 'x')
        # Checking if it is last block
        if i + blockSize + 1 > realLength:
            while len(decryptedBlock) < lastChunkSize:
                decryptedBlock = '0' + decryptedBlock
        # Make length dividable by 2
        else:
            while len(decryptedBlock) % int(blockSize/4) != 0:
                decryptedBlock = '0' + decryptedBlock
        # Add to new idat data
        newIdat += decryptedBlock
        i += blockSize
    return newIdat


# Function encrypt data got in imageData argument
def encryptData(imageData, blockSize, realLength, n, e):
    newIdat = ""
    i = 0
    # While there is still data
    while i < realLength:
        # Get correct blocks
        if (i + blockSize) > realLength:
            block = imageData[i:i + (realLength - i)]
        else:
            block = imageData[i:i + blockSize]
        if len(block) < blockSize:
            lastChunkSize = len(block)
        # Change it into int
        blockInInt = int(block, 16)
        # Encrypt number
        encryptedNumber = encryptNumber(blockInInt, e, n)
        # And make it hex
        encryptedBlock = format(encryptedNumber, 'x')
        # Align size to blockSize length
        while len(encryptedBlock) % blockSize != 0:
            encryptedBlock = '0' + encryptedBlock
        # And add it into new idat data
        newIdat += encryptedBlock
        i += blockSize
    return newIdat


def encryptDataCBC(imageData, blockSize, realLength, n, e):
    newIdat = ""
    i = 0
    # While there is still data
    while i < realLength:
        # Get correct blocks
        if (i + blockSize) > realLength:
            block = imageData[i:i + (realLength - i)]
        else:
            block = imageData[i:i + blockSize]
        if len(block) < blockSize:
            lastChunkSize = len(block)
        # Change it into int
        blockInInt = int(block, 16)
        # Encrypt first block
        if (i==0):
            encryptedNumber = encryptNumber(blockInInt, e, n)
        # Encrypt other blocks
        if(i!=0):
            CBCblock = blockInInt ^ LastBlock
            encryptedNumber = encryptNumber(CBCblock, e, n)
        # Remember last encrypted block
        LastBlock=encryptedNumber
        # And make encryptedNumber hex
        encryptedBlock = format(encryptedNumber, 'x')
        # Align size to blockSize length
        while len(encryptedBlock) % blockSize != 0:
            encryptedBlock = '0' + encryptedBlock
        # And add it into new idat data
        newIdat += encryptedBlock
        i += blockSize
    return newIdat

def decryptDataCBC(imageData, blockSize, realLength, d, n):
    newIdat = ""
    i = 0
    # Encrypted data is 4 times bigger than not encrypted
    blockSize *= 4
    # While there is still data
    while i < realLength:
        # Get block in correct size
        block = imageData[i:i + blockSize]
        # Change into int
        decBlockInInt = int(block, 16)
        # Decrypt first number
        if (i==0):
            decryptedNumber = decryptNumber(decBlockInInt, d, n)
        # Encrypt other blocks
        if (i!=0):
            decryptedNumber = decryptNumber(decBlockInInt, d, n)
            decryptedNumber = decryptedNumber ^ LastBlock
        # Remember last encrypted block
        #LastBlock = decryptedNumber
        LastBlock = decBlockInInt
        # And make it hex
        decryptedBlock = format(decryptedNumber, 'x')
        # Checking if it is last block
        if i + blockSize + 1 > realLength:
            while len(decryptedBlock) < lastChunkSize:
                decryptedBlock = '0' + decryptedBlock
        # Make length dividable by 2
        else:
            while len(decryptedBlock) % int(blockSize/4) != 0:
                decryptedBlock = '0' + decryptedBlock
        # Add to new idat data
        newIdat += decryptedBlock
        i += blockSize
    return newIdat

def encryptDataLib(imageData, blockSize, realLength, n, e):
    newIdat = ""
    i = 0
    # While there is still data
    while i < realLength:
        # Get correct blocks
        if (i + blockSize) > realLength:
            block = imageData[i:i + (realLength - i)]
        else:
            block = imageData[i:i + blockSize]
        if len(block) < blockSize:
            lastChunkSize = len(block)
        # Change it into int
        blockInInt = bytes.fromhex(block)
        # Encrypt number
        key = PublicKey(n, e)
        encryptedNumber = rsa.encrypt(blockInInt,key )
        # And make it hex
        encryptedBlock = encryptedNumber.hex() ##
        # Align size to blockSize length
        while len(encryptedBlock) % blockSize != 0:
            encryptedBlock = '0' + encryptedBlock
        # And add it into new idat data
        newIdat += encryptedBlock
        i += blockSize
    return newIdat

def decryptDataLib(imageData, blockSize, realLength, d, n):
    newIdat = ""
    i = 0
    # Encrypted data is 4 times bigger than not encrypted
    blockSize *= 4
    # While there is still data
    while i < realLength:
        # Get block in correct size
        block = imageData[i:i + blockSize]
        # Change into int
        blockInInt = bytes.fromhex(block)
        # Decrypt number
        key = PrivateKey(n, e, d, p, q)
        decryptedNumber = rsa.decrypt(blockInInt, key)
        # And make it hex
        decryptedBlock = decryptedNumber.hex()
        # Checking if it is last block
        if i + blockSize + 1 > realLength:
            while len(decryptedBlock) < lastChunkSize:
                decryptedBlock = '0' + decryptedBlock
        # Make length dividable by 2
        else:
            while len(decryptedBlock) % int(blockSize/4) != 0:
                decryptedBlock = '0' + decryptedBlock
        # Add to new idat data
        newIdat += decryptedBlock
        i += blockSize
    return newIdat

# Method is the type of encryption 1: own RSA encryption, 2: library RSA encryption
def encodePicture(input, output, method, n, e):
    old_png = open(input, "rb")
    new_png = open(output, 'wb')
    for i in range(8):
        signature = old_png.read(1)
        new_png.write(signature)
    height = 0
    width = 0
    colortype = ""
    chunk_length = old_png.read(4)
    chunk_length2 = int.from_bytes(chunk_length, byteorder='big')
    chunk_type = old_png.read(4)
    chunk_type2 = bytearray.fromhex(chunk_type.hex()).decode()
    chunk_data = old_png.read(chunk_length2)
    crc = old_png.read(4)
    while chunk_data:
        if chunk_type2.upper() in {"IDAT"}:
            #chunkLengthDec = chunk_length2
            chunkLengthDec = int.from_bytes(chunk_length, byteorder='big')
            realLength = 2 * chunkLengthDec
            blockSize = int(primeBitsCount/8)
            if method == 1:
                newIdatHex = encryptData(chunk_data.hex(), blockSize, realLength, n, e)
            #data = chunk_type
            elif method == 2:
                newIdatHex = encryptDataLib(chunk_data.hex(), blockSize, realLength, n, e)
            elif method == 3:
                newIdatHex = encryptDataCBC(chunk_data.hex(), blockSize, realLength, n, e)
            else:
                newIdatHex = encryptData(chunk_data.hex(), blockSize, realLength, n, e)
            newLength = int(len(newIdatHex)/2)
            newHexIdatLength = format(newLength, 'x')
            while len(newHexIdatLength) % 8 != 0:
                newHexIdatLength = '0' + newHexIdatLength

            new_png.write(bytes.fromhex(newHexIdatLength))
            new_png.write(chunk_type)
            calc_crc = zlib.crc32(chunk_type + bytes.fromhex(newIdatHex)) & 0xffffffff
            calc_crc = struct.pack('!I', calc_crc)
            new_png.write(bytes.fromhex(newIdatHex))
            #new_png.write(crc)
            new_png.write(calc_crc)
        elif chunk_type2.upper() == "IEND":
            print("ELO")
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
    #new_png.write(chunk_data)
    new_png.write(crc)
    new_png.write(bytes(lastChunkSize))
    new_png.close()
    old_png.close()


# Method is the type of decryption 1: own RSA decryption, 2: library RSA decryption
def decodePicture(input, output, method, n, d, e, p, q):
    old_png = open(input, "rb")
    new_png = open(output, 'wb')
    for i in range(8):
        signature = old_png.read(1)
        new_png.write(signature)
    height = 0
    width = 0
    colortype = ""
    chunk_length = old_png.read(4)
    chunk_length2 = int.from_bytes(chunk_length, byteorder='big')
    chunk_type = old_png.read(4)
    chunk_type2 = bytearray.fromhex(chunk_type.hex()).decode()
    chunk_data = old_png.read(chunk_length2)
    crc = old_png.read(4)
    while chunk_data:
        if chunk_type2.upper() in {"IDAT"}:
            #chunkLengthDec = chunk_length2
            chunkLengthDec = int.from_bytes(chunk_length, byteorder='big')
            realLength = 2 * chunkLengthDec
            blockSize = int(primeBitsCount/8)
            if method == 1:
                newIdatHex = decryptData(chunk_data.hex(), blockSize, realLength, d, n)
            #data = chunk_type
            elif method == 2:
                newIdatHex = decryptDataLib(chunk_data.hex(), blockSize, realLength, d, n)
            elif method == 3:
                newIdatHex = decryptDataCBC(chunk_data.hex(), blockSize, realLength, d, n)
            else:
                newIdatHex = decryptData(chunk_data.hex(), blockSize, realLength, n, e, d)
            newLength = int(len(newIdatHex)/2)
            newHexIdatLength = format(newLength, 'x')
            while len(newHexIdatLength) % 8 != 0:
                newHexIdatLength = '0' + newHexIdatLength
            # print('orginal length hex')
            # print(realLength)
            # print("from chunkt")
            # print(chunk_length2)
            # print('encrypted lenght')
            # print(newLength)
            # print("Równe hex length? Old: " + str(len(chunk_data.hex())) + "||New: " + str(len(newIdatHex)))
            # print(len(chunk_data.hex()) == len(newIdatHex))
            # bytesData = bytes.fromhex(newIdatHex)
            # print("Równe bytes length? Old: "+str(len(chunk_data))+"||New: " + str(len(bytesData)))
            # print(len(chunk_data) == len(bytesData))

            new_png.write(bytes.fromhex(newHexIdatLength))
            #new_png.write(chunk_length)
            new_png.write(chunk_type)
            #calc_crc = zlib.crc32(chunk_type + bytes.fromhex(newIdatHex)) & 0xffffffff
            #calc_crc = struct.pack('!I', calc_crc)
            new_png.write(bytes.fromhex(newIdatHex))
            new_png.write(crc)
            #new_png.write(calc_crc)
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
    #new_png.write(chunk_data)
    new_png.write(crc)
    new_png.close()
    old_png.close()



start = time.time()
keys = generateRSA()
end = time.time()
print("Public key: " + str(keys["public"]) + "\nPrivate key: " + str(keys["private"]))
print("Finding key time: " + str(end - start))
#print("::::!TESTS!::::")
#print(7 == encryptNumber(123, 7, 143))  # For 123 number and public key (7,143) should be 7
#print(123 == encryptNumber(7, 103, 143))  # For 7 number and private key (103, 143) should be 123
filein = "PNGFile4.png"
fileout = "out.png"
decodeOut = "decoded.png"

n = keys["public"]["n"]
e = keys["public"]["e"]
d = keys["private"]["d"]
p = keys["private"]["p"]
q = keys["private"]["q"]

encodePicture(filein, fileout, 1, n, e)
decodePicture(fileout, decodeOut, 1, n, d, e, p, q)

tryb = int(input("Wybierz tryb działania "
                 "\nDostepne opcje: \n0 - dokodowanie pliku, \n1 - anonimizacja pliku, \n2 - FFT, "
                 "\n3 - wyswietl zdjecie,\n4 - koduj obraz,\n5 - dekoduj obraz,\n9 - wyjscie \nWybor: "))

while (tryb != 9):
    if tryb == 0:
        input_file = input("Podaj nazwe pliku wejsciowego PNG: ")
        output_file = input("Podaj nazwe pliku wyjsciowego, gdzie zapisane zostana zdekodowane chunki: ")
        dekodowanie(input_file, output_file)
    elif tryb == 1:
        input_file = input("Podaj nazwe pliku wejsciowego PNG: ")
        wyjscie = input('Podaj nazwe pliku wyjsciowego png po anonimizacji: ')
        anonimizacja(input_file, wyjscie)
    elif tryb == 2:
        transform_file = input("Podaj nazwe pliku którego transformacji należy dokonać: ")
        doFourierTransform(transform_file)
    elif tryb == 3:
        picture_to_show = input("Podaj nazwe pliku png do wyswietlenia: ")
        showImage(picture_to_show)
    elif tryb == 4:
        picture_to_encode = input("Podaj nazwe pliku do zakodowania: ")
        picture_after_encode = input("Podaj nazwe pliku wynikowego: ")
        start = time.time()
        encodePicture(picture_to_encode, picture_after_encode, 1, n, e)
        end = time.time()
        print("Encoding time: " + str(end - start))
    elif tryb == 5:
        picture_to_decode = input("Podaj nazwe pliku do zdekodowania: ")
        picture_after_decode = input("Podaj nazwe pliku wynikowego: ")
        start = time.time()
        decodePicture(picture_to_decode, picture_after_decode , 1, n, d, e, p, q)
        end = time.time()
        print("Decoding time: " + str(end - start))
    else:
        print("Błędny tryb! Wybierz z listy prawidłowy!!")
    tryb = int(input("Wybierz tryb działania "
                     "\nDostepne opcje: \n0 - dokodowanie pliku, \n1 - anonimizacja pliku, \n2 - FFT, "
                     "\n3 - wyswietl zdjecie, \n9 - wyjscie \nWybor: "))