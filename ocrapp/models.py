from django.db import models


# Create your models here.
class Recon(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    ori_photo = models.ImageField(upload_to='ori_photos/%Y/%m/%d/',blank=True, null=True)
    gray_photo = models.ImageField(upload_to='gray_photos/%Y/%m/%d/',blank=True, null=True)
    binary_photo = models.ImageField(upload_to='binary_photos/%Y/%m/%d/',blank=True, null=True)
    morpho_photo = models.ImageField(upload_to='morpho_photos/%Y/%m/%d/',blank=True, null=True)
    smoothed_photo = models.ImageField(upload_to='smoothed_photos/%Y/%m/%d/', blank=True, null=True)
    final_photo = models.ImageField(upload_to='final_photos/%Y/%m/%d/', blank=True, null=True)
    context = models.CharField(max_length=2000)

