@echo off
REM Automated setup for Tailwind CSS, Chart.js, and Font Awesome
REM Run this script from your project root directory

REM Install dependencies
npm install tailwindcss postcss autoprefixer chart.js @fortawesome/fontawesome-free

REM Build Tailwind CSS
npx tailwindcss -i ./src/styles.css -o ./dist/tailwind.css --minify

echo Setup complete. You can now use local assets in your frontend.
pause
