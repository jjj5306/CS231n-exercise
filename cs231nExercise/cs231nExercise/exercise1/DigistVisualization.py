# 시각화 모듈
import matplotlib.pyplot as plt

# Import datasets and performance metrics
from sklearn import datasets, metrics

# The handwritten digits dataset (8X8 image)
digits = datasets.load_digits()

# Target
images_and_labels = list(zip(digits.images, digits.target))

# Visualization of 4 images
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index+1) # 2 X 4의 subplot 위치를 할당하고, 각 image를 할당해서 그림
    plt.axis('off') # 축 없음
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest') # image 파일을 시각화 하는 함수. 타입을 흑백으로
    plt.title('Training: %i' % label) # title 설정

# 그래프 출력
plt.show()