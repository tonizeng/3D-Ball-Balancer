#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// Create the PWM driver object
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// Servo pulse range (calibrate if needed)
#define SERVOMIN  150  // Min pulse length out of 4096
#define SERVOMAX  600  // Max pulse length out of 4096

#define ANGLEMIN  0
#define ANGLEMAX  360

// Servo channels
#define SERVO1 0
#define SERVO2 1
#define SERVO3 2

int angleToPulse(int angle) 
{
  return map(angle, ANGLEMIN, ANGLEMAX, SERVOMIN, SERVOMAX);
}

void moveMotors(int angle1, int angle2, int angle3)
{
  // Offset for each motor
      angle1+=60;
      angle2+=80;
      angle3+=20;
  
      // Constrain angles to valid range
      angle1 = constrain(angle1, 60, 180);
      angle2 = constrain(angle2, 80, 200);
      angle3 = constrain(angle3, 20, 140);
  
      // Convert to pulse width and set servos
      pwm.setPWM(SERVO1, 0, angleToPulse(angle1));
      pwm.setPWM(SERVO2, 0, angleToPulse(angle2));
      pwm.setPWM(SERVO3, 0, angleToPulse(angle3));
  
      Serial.print("Servo 1: "); Serial.print(angle1-60);
      Serial.print(" | Servo 2: "); Serial.print(angle2-80);
      Serial.print(" | Servo 3: "); Serial.println(angle3-20); 
}

void setup() 
{
  Serial.begin(9600);  
  pwm.begin();
  pwm.setPWMFreq(50);   // Servos use ~50 Hz refresh rate
  moveMotors(0, 0, 0);  // Move to default position
  
  delay(10);
  Serial.println("Enter three angles between 0 and 120 seperated by 1 space:");
}

void loop() 
{
  // Check if serial data is available
  if (Serial.available()) {
    // Read the whole line (e.g., "60 70 80")
    String input = Serial.readStringUntil('\n');
    input.trim();  // Remove newline or spaces
  
    // Split into three parts
    int angle1, angle2, angle3;
    int numScanned = sscanf(input.c_str(), "%d %d %d", &angle1, &angle2, &angle3);

    // Test Pattern
    if(angle1 == 360 && angle2 == 360 && angle3 == 360)
    {
      moveMotors(0, 0, 0);
      delay(500);

      moveMotors(120, 0, 0);
      delay(500);

      moveMotors(0, 120, 0);
      delay(500);

      moveMotors(0, 0, 120);
      delay(500);

      moveMotors(120, 120, 0);
      delay(500);

      moveMotors(0, 120, 120);
      delay(500);

      moveMotors(120, 0, 120);
      delay(500);

      moveMotors(0, 0, 0);
      delay(500);
            
      moveMotors(40, 40, 40);
      delay(500);

      moveMotors(80, 80, 80);
      delay(500);

      moveMotors(120, 120, 120);
      delay(500);

      Serial.println("Test Sequence Complete");
    }
    else 
    {
      // Use the regular 3 input angles
      moveMotors(angle1, angle2, angle3);
    }
  }
}
