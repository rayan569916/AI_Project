import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ChatSectionComponent } from './chat.section.component';

describe('ChatSectionComponent', () => {
  let component: ChatSectionComponent;
  let fixture: ComponentFixture<ChatSectionComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ChatSectionComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ChatSectionComponent);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
