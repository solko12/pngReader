tryb = int(input("Wybierz tryb działania ( 0 - dokodowanie pliku , 1 - anonimizacja pliku) : "))

if (tryb == 0):
    file = open("PNGFile.png", "rb")
    print("::PNG SIGNATURE::")


    for i in range(8):
        signature = file.read(1)
        print(signature.hex())

    chunk_length = int(file.read(4).hex(), 16)
    chunk_type = bytearray.fromhex(file.read(4).hex()).decode()
    chunk_data = file.read(chunk_length)
    crc = file.read(4)

    data_for_FFT = ""
    plik = open('wyjsciowy.txt','w')

    while chunk_data:
        plik.write("\n\n::NEXT CHUNK::")
        plik.write("\nChunk length: ")
        plik.write(str(chunk_length))
        plik.write("\nChunk type: ")
        plik.write(str(chunk_type))

        print("\n::NEXT CHUNK::")
        print("Chunk length: ", chunk_length)
        print("Chunk type: ", chunk_type)
        filter_method = 0;

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
                print("Chunk IDAT data: ", variable)
                plik.write("\nChunk IDAT data: "+ str(variable))
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

            if chunk_type.upper() == "TEXT":
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

        chunk_length = int(file.read(4).hex(), 16)
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
if (tryb == 1):
   file = open("PNGFile.png", "rb")
   plik = open('PNGFileAnim.png','w')

   for i in range(8):
         signature = file.read(1)
         plik.write(str(signature))



   chunk_length = file.read(4)
   chunk_length2 = int(chunk_length.hex(), 16)
   chunk_type = file.read(4)
   chunk_type2 = bytearray.fromhex(chunk_type.hex()).decode()
   chunk_data = file.read(chunk_length2)
   crc = file.read(4)

   while chunk_data:

        if chunk_type2.upper() in {"IHDR","IDAT","PLTE","IEND"}:
       #    print("oo")
           plik.write(str(chunk_length))
           plik.write(str(chunk_type))
           plik.write(str(chunk_data))
           plik.write(str(crc))


        #data_for_FFT = data_for_FFT+variable

        chunk_length = file.read(4)
        chunk_length2 = int(chunk_length.hex(), 16)
        chunk_type = file.read(4)
        chunk_type2 = bytearray.fromhex(chunk_type.hex()).decode()
        chunk_data = file.read(chunk_length2)
        crc = file.read(4)
   plik.write(str(chunk_length))
   plik.write(str(chunk_type))
   plik.write(str(chunk_data))
   plik.write(str(crc))

