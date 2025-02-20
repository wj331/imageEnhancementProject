import { Component } from '@angular/core';
import { ImageUploadComponent } from './image-upload/image-upload.component';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatButtonModule } from '@angular/material/button';
import { MatSidenavModule } from '@angular/material/sidenav';
import { MatIconModule } from '@angular/material/icon';
import { MatSnackBarModule } from '@angular/material/snack-bar';
import { OverlayContainer } from '@angular/cdk/overlay';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    ImageUploadComponent,
    MatToolbarModule,
    MatButtonModule,
    MatSidenavModule,
    MatIconModule,
    MatSnackBarModule
  ],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'image-enhancement-app';
  darkMode = false;

  constructor(private overlayContainer: OverlayContainer) {}

  toggleDarkMode() {
    this.darkMode = !this.darkMode;
    const overlayContainerClasses = this.overlayContainer.getContainerElement().classList;

    if (this.darkMode) {
      document.body.classList.add('dark-theme');
      overlayContainerClasses.add('dark-theme');
    } else {
      document.body.classList.remove('dark-theme');
      overlayContainerClasses.remove('dark-theme');
    }
  }
}
