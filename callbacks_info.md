
# چیستی، کاربرد و پیاده‌سازی Callbackهای سفارشی در TensorFlow/Keras

## مقدمه
در فرآیند آموزش مدل‌های یادگیری عمیق، نیاز به کنترل دقیق و سفارشی‌سازی رفتار مدل در مراحل مختلف آموزش، ارزیابی و پیش‌بینی وجود دارد. Callbackها در TensorFlow/Keras ابزارهای قدرتمندی هستند که این امکان را فراهم می‌کنند. این مقاله به بررسی مفهوم Callbackهای سفارشی، دلایل پیدایش، کاربردها، مزایا و معایب، و نحوه پیاده‌سازی آن‌ها با مثال‌های عملی و کاربردی می‌پردازد. همچنین، با استناد به منابع معتبر مانند مستندات TensorFlow و Keras، این موضوع را به‌صورت علمی و جامع تحلیل می‌کنیم.

---

## چیستی Callbackها
Callbackها توابعی هستند که در نقاط خاصی از فرآیند آموزش، ارزیابی یا پیش‌بینی مدل (مانند شروع/پایان یک epoch، شروع/پایان یک batch، یا پایان کل فرآیند آموزش) به‌صورت خودکار فراخوانی می‌شوند. آن‌ها به توسعه‌دهندگان اجازه می‌دهند تا رفتار مدل را نظارت، کنترل یا تغییر دهند. Callbackهای سفارشی با ارث‌بری از کلاس پایه `tf.keras.callbacks.Callback` پیاده‌سازی می‌شوند و امکان تعریف رفتارهای خاص برای نیازهای پروژه را فراهم می‌کنند.

---

## دلایل پیدایش Callbackها
- **نیاز به انعطاف‌پذیری:** نظارت و کنترل دقیق در مراحل مختلف آموزش.
- **اتوماسیون فرآیندها:** کاهش دخالت دستی در ذخیره مدل یا ثبت لاگ‌ها.
- **رفع مشکلات خاص:** جلوگیری از بیش‌برازش یا مدیریت مقادیر NaN.
- **پشتیبانی صنعتی:** ثبت لاگ‌ها و ذخیره مدل در سرورهای ابری.

---

## نحوه عملکرد و ترتیب مراحل
Callbackها در نقاط زیر فراخوانی می‌شوند:

### 1. جهانی
- `on_train_begin`, `on_train_end`
- `on_evaluate_begin`, `on_evaluate_end`
- `on_predict_begin`, `on_predict_end`

### 2. سطح epoch
- `on_epoch_begin`, `on_epoch_end`

### 3. سطح batch
- `on_train_batch_begin`, `on_train_batch_end`
- `on_test_batch_begin`, `on_test_batch_end`
- `on_predict_batch_begin`, `on_predict_batch_end`

هر متد به `logs` (حاوی معیارهای عملکرد) و از طریق `self.model` به مدل دسترسی دارد.

---

## مزایا
| مزیت             | توضیح |
|------------------|-------|
| انعطاف‌پذیری بالا | تغییر رفتار مدل مطابق نیاز پروژه |
| اتوماسیون         | کاهش دخالت دستی در فرآیندها |
| نظارت و کنترل     | توقف آموزش یا تغییر پارامترها بر اساس معیارها |
| کاربرد صنعتی     | ذخیره در سرورهای ابری یا ثبت لاگ پیشرفته |

## معایب
| عیب               | توضیح |
|-------------------|-------|
| پیچیدگی پیاده‌سازی | نیاز به دانش عمیق از Keras و TensorFlow |
| افزایش زمان محاسبات | Callbackهای پیچیده می‌توانند آموزش را کند کنند |
| خطای احتمالی       | پیاده‌سازی نادرست باعث خطا می‌شود |

---

## مقایسه با سایر روش‌ها
| روش                  | مزایا                                      | معایب                              | کاربرد |
|----------------------|--------------------------------------------|--------------------------------------|--------|
| Callback سفارشی      | انعطاف‌پذیری بالا                          | پیچیدگی، احتمال خطا                 | نیازهای خاص |
| Callback داخلی Keras | آماده، پایدار                              | محدود به عملکرد از پیش تعریف‌شده    | پروژه‌های استاندارد |
| دخالت دستی           | کنترل کامل                                 | زمان‌بر، خطای انسانی                 | پروژه‌های کوچک |
| ابزار خارجی (MLflow) | قابلیت‌های پیشرفته                         | نیاز به تنظیمات اضافی                | پروژه‌های صنعتی |

---

## چه موقع استفاده کنیم؟
**استفاده کنید:**
- نیاز به رفتار خاص (تغییر نرخ یادگیری، اعلان به سرور)
- پروژه‌های صنعتی با نیاز به لاگ‌گیری پیشرفته
- جلوگیری از بیش‌برازش یا نظارت بر معیار خاص

**استفاده نکنید:**
- پروژه‌های ساده با کافی بودن Callbackهای داخلی
- کمبود دانش فنی لازم
- محدودیت شدید زمان محاسبات

---

## مثال عملی: Callback سفارشی
```python
import os
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

class CustomAccuracyCallback(Callback):
    def __init__(self, accuracy_threshold=0.98, checkpoint_path=None):
        super().__init__()
        self.accuracy_threshold = accuracy_threshold
        self.checkpoint_path = checkpoint_path

    def on_epoch_end(self, epoch, logs=None):
        current_accuracy = logs.get('val_accuracy')
        if current_accuracy is not None and current_accuracy >= self.accuracy_threshold:
            print(f"\nReached {self.accuracy_threshold*100}% validation accuracy. Stopping training!")
            self.model.stop_training = True
            if self.checkpoint_path:
                self.model.save(self.checkpoint_path)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

checkpoint_path = os.path.join(checkpoint_dir, 'best_model.keras')

model.fit(x_train, y_train, 
          epochs=20, 
          validation_data=(x_test, y_test), 
          callbacks=[CustomAccuracyCallback(0.98, checkpoint_path)])
````

---

## کاربردهای صنعتی

| صنعت            | کاربرد Callback سفارشی                   | مثال                        |
| --------------- | ---------------------------------------- | --------------------------- |
| پزشکی           | توقف آموزش در رسیدن به دقت خاص           | مدل تشخیص سرطان پوست        |
| مالی            | تغییر نرخ یادگیری بر اساس نوسانات بازار  | کاهش LR در صورت افزایش loss |
| خودرو           | ثبت لاگ پیشرفته برای سیستم‌های خودران    | ارسال لاگ به سرور ابری      |
| تجارت الکترونیک | ذخیره مدل‌های بهینه برای پیشنهاد محصولات | ذخیره مدل در AWS            |

---

## پیشنهادها

1. بررسی Callbackهای داخلی پیش از پیاده‌سازی سفارشی.
2. بهینه‌سازی منابع در Callbackهای پیچیده.
3. تست روی دیتاست کوچک قبل از اجرا در مقیاس بزرگ.
4. ترکیب چند Callback در `model.fit`.
5. استفاده از TensorBoard برای نظارت بصری.

---

## نتیجه‌گیری

Callbackهای سفارشی در TensorFlow/Keras ابزارهایی قدرتمند برای کنترل و سفارشی‌سازی فرآیند آموزش هستند. این ابزارها انعطاف‌پذیری بالایی دارند و در کاربردهای صنعتی می‌توانند نقش کلیدی ایفا کنند. با پیاده‌سازی صحیح و استفاده هوشمندانه، می‌توان فرآیند آموزش را بهینه و نتایج بهتری کسب کرد.

---

## منابع

* [مستندات رسمی TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback)
* [راهنمای نوشتن Callbackهای سفارشی](https://www.tensorflow.org/guide/keras/custom_callback)
* [Keras Callbacks API](https://keras.io/api/callbacks/)
* مقالات Medium و StackAbuse درباره Callbackهای Keras
