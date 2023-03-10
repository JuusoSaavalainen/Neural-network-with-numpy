def convert_to_csv(image, label, out, n):
    '''
    Source : https://pjreddie.com/projects/mnist-in-csv/
    '''
    img = open(image, "rb")
    lab = open(label, "rb")
    output = open(out, "w")

    img.read(16)  # 16 bytes
    lab.read(8)  # 8 bytes
    images = []

    for i in range(n):
        image = [ord(lab.read(1))]  # lab
        for j in range(784):
            image.append(ord(img.read(1)))  # img
        images.append(image)

    for image in images:  # converting to csv format
        output.write(",".join(map(str, image)) + "\n")

    img.close()
    lab.close()
    output.close()


convert_to_csv("src/data/train-images.idx3-ubyte", "src/data/train-labels.idx1-ubyte",
               "src/data/mnist_train.csv", 60000)

convert_to_csv("src/data/t10k-images.idx3-ubyte", "src/data/t10k-labels.idx1-ubyte",
               "src/data/mnist_test.csv", 10000)
