%% Initialization
clc
clear all;
close all
syms t1 tf xf yf x1 y1 float t
t1 = sym('t1', 'real');
tf = sym('tf', 'real');
xf = sym('xf', 'real');
yf = sym('yf', 'real');
x1 = sym('x1', 'real');
y1 = sym('y1', 'real');
t =  sym('t' , 'real');
xf = 0;
yf = 0.1;
x1 = 0.2;
y1 = 0.3;
tf = 1
%% Finding the time to arrive at point x1 and y1
A2 = expand(xf*(120*t1^5 - 300*t1^4 + 200*t1^3) -20*x1);
A22 = expand(yf*(120*t1^5 - 300*t1^4 + 200*t1^3) -20*y1);
A1 = expand(xf*(300*t1^5 - 1200*t1^4 + 1600*t1^3) + t1^2*(-720*xf + 120*x1) - x1*(300*t1 - 200));
A11 = expand(yf*(300*t1^5 - 1200*t1^4 + 1600*t1^3) + t1^2*(-720*yf + 120*y1) - y1*(300*t1 - 200));
A3 = 60*t1^7 - 210*t1^6 + 240*t1^5 - 90*t1^4;
A4 = 60*t1^3 - 30*t1^2 - 30*t1^4;
Result = expand(A3*A2*A2 + A3*A22*A22 + (t1^3)*A1*A2*A4 + (t1^3)*A11*A22*A4);
%numerical solution
solvet = vpasolve(Result, t1)

real_roots = solvet(imag(solvet)==0); % filter out only real roots
T1 = real_roots(real_roots > 0 & real_roots < 1) % filter out roots between 0 and 1

%% X-Position Equations as a function of time

pi1 = 1/(tf^5 * t1^5 * ((1-t1)^5)) * A2;
c1 = 1/(tf^5 * t1^2 * ((1-t1)^5)) * A1;
star1(t) = t1^4 * (15*t^4 - 30*t^3)+t1^3 * (80*t^3-30*t^4)-60*t^3*t1^2+30*t^4*t1-6*t^5;
star2(t) = 15*t^4 - 10*t^3 - 6*t^5;

 x_negative(t,t1) = (tf^5)/720 * ((pi1*star1)+ c1 *star2);
 x_positive(t,t1) =x_negative(t,t1) + pi1* (t1^5 * ((t-t1)^5))/120;

Position_X(t,t1) =  heaviside(t1 - t) * x_negative(t,t1) +heaviside(t - t1) * x_positive(t,t1) ;

figure
subplot(2,2,1)
 fplot(Position_X(t,T1), [0 tf])
title("X POSITION")

%% Y-Position Equations as a function of time
pi2 = 1/(tf^5 * t1^5 * ((1-t1)^5)) * A22;
c2 = 1/(tf^5 * t1^2 * ((1-t1)^5)) * A11;

 y_negative(t,t1) = (tf^5)/720 * ((pi2*star1)+ c2 *star2);
 y_positive(t,t1) = y_negative(t,t1) + pi2* (t1^5 * ((t-t1)^5))/120;

Position_Y(t,t1) =  heaviside(t1 - t) * y_negative(t,t1) +heaviside(t - t1) * y_positive(t,t1) ;

subplot(2,2,2)
fplot(Position_Y(t,T1), [0 tf])
title("Y POSITION")


%% X-Velocity Equations as a function of time

Vx_negative = tf^5/720*(pi1*(t1^4*(60*t^3-90*t^2)+t1^3*(240*t^2-120*t^3)-180*t^2*t1^2+120*t^3*t1-30*t^4)+c1*(60*t^3-30*t^2-30*t^4));
Vx_positive = Vx_negative + pi1/24*(t1^5*(t-t1)^4);

Velocity_X(t,t1) = heaviside(t1 - t) * Vx_negative + heaviside(t - t1) * Vx_positive;

subplot(2,2,3)
 fplot(Velocity_X(t,T1) , [0 tf])
title("X VELOCITY")

%% Y-Velocity Equations as a function of time

Vy_negative = tf^5/720*(pi2*(t1^4*(60*t^3-90*t^2)+t1^3*(240*t^2-120*t^3)-180*t^2*t1^2+120*t^3*t1-30*t^4)+c2*(60*t^3-30*t^2-30*t^4));
Vy_positive = Vy_negative + pi2/24*(t1^5*(t-t1)^4);

Velocity_Y(t,t1) = heaviside(t1 - t) * Vy_negative + heaviside(t - t1) * Vy_positive;

subplot(2,2,4)

fplot(Velocity_Y(t,T1) , [0 tf])
title("Y VELOCITY")

