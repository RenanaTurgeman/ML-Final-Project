import matplotlib.pyplot as plt

categories = ['KNN', 'Logistic Regression', 'Decision Tree']
accuracy_color = [0.32,  0.3465,  0.2590]
accuracy_gray = [0.28, 0.2645, 0.2152]

fig, ax = plt.subplots()
width = 0.35
x = range(len(categories))

bars1 = ax.bar(x, accuracy_color, width, label='Color Images')
bars2 = ax.bar([p + width for p in x], accuracy_gray, width, label='Grayscale Images')

ax.set_xlabel('Categories')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy by Image Type')
ax.set_xticks([p + width/2 for p in x])
ax.set_xticklabels(categories)
ax.legend()

plt.show()
