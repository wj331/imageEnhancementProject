import { Component } from '@angular/core';
import { ImageUploadComponent } from './image-upload/image-upload.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [ImageUploadComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'image-enhancement-app';
}
