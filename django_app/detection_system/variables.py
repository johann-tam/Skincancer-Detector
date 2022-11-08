"""
Original author(s): Johann Tammen
Modified by: Linus Ivarsson, Linus Åberg

File purpose: Store important variables of the detection_system
"""
LABEL_DIRECTORY = (["Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec)",
                        "Basal cell carcinoma (bcc)", "Benign keratosis-like lesions (bkl)", "Dermatofibroma (df)",
                        "Melanoma (mel)", "Melanocytic nevi (nv)", "Vascular lesions (vascs (multiple))"])

LINK_DIRECTORY = (["https://www.skincancer.org/skin-cancer-information/actinic-keratosis/",
                        "https://www.skincancer.org/skin-cancer-information/basal-cell-carcinoma/",
                        "https://dermnetnz.org/topics/seborrhoeic-keratosis",
                        "https://dermnetnz.org/topics/dermatofibroma",
                        "https://www.skincancer.org/skin-cancer-information/melanoma/",
                        "https://dermnetnz.org/topics/melanocytic-naevus",
                        "https://dermnetnz.org/topics/vascular-skin-problems"])

IMAGEWIDTH = 28  # Image size used for resizing an input image into what the ML model can accept
IMAGEHEIGHT = 28  # Image size used for resizing an input image into what the ML model can accept
AMTCOLUMNS = 785  # 28x28x1 + label

DTYPE = 'float32'
CSVRANGE = 255

COLUMNSAXIS = 1
PADDING = 2
CHANNELS = 3
CHANNELPOSITION = -1

PADDEDWIDTH = 32
PADDEDHEIGHT = 32

OUTPUTNEURONS = 7
EPOCHS = 5
BATCHSIZE = 32
