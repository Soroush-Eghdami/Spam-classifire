# Initial prediction table.

import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Prediction',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('text_preview', models.CharField(max_length=280)),
                ('text_length', models.PositiveIntegerField(default=0)),
                ('label', models.CharField(choices=[('spam', 'Spam'), ('ham', 'Ham')], db_index=True, max_length=8)),
                ('spam_probability', models.FloatField()),
                ('threshold', models.FloatField(default=0.5)),
                ('top_words', models.JSONField(blank=True, default=list)),
                ('model_version', models.CharField(default='unknown', max_length=32)),
                ('source', models.CharField(choices=[('single', 'Single message'), ('batch', 'Batch upload')], db_index=True, default='single', max_length=16)),
                ('created_at', models.DateTimeField(db_index=True, default=django.utils.timezone.now)),
            ],
            options={
                'ordering': ['-created_at', '-id'],
                'indexes': [models.Index(fields=['label', '-created_at'], name='pred_label_created_idx'), models.Index(fields=['source', '-created_at'], name='pred_source_created_idx')],
            },
        ),
    ]
