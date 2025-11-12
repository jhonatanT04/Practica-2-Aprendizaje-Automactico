import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { SupermercadoComponent } from './supermercado/supermercado.component';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet,SupermercadoComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent {
  title = 'appsupermercado';
}
