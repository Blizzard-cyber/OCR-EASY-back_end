from rest_framework import serializers

from .models import Recon


class rcoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Recon
        fields = '__all__'
