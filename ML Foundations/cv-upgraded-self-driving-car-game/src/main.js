import { getRandomInt } from "./utils/utils.js";
import { getAngleDeg } from "./utils/mediapipe.js";
import { Car } from "./components/car.js";
import { Road } from "./components/road.js";
import { NeuralNetwork } from "./components/network.js";
import { Visualizer } from "./components/visualizer.js";

var twoPlayers = false;

localStorage.setItem("twoPlayers", JSON.stringify(twoPlayers));

// if (JSON.parse(localStorage.getItem("twoPlayers")) == false)

const carCanvas = document.getElementById("carCanvas");
carCanvas.width = 200;
const carCtx = carCanvas.getContext("2d");
const networkCanvas = document.getElementById("networkCanvas");
networkCanvas.width = 300;
const numTrafficObstacles = 10;
const networkCtx = networkCanvas.getContext("2d");

const road = new Road(carCanvas.width / 2, carCanvas.width * 0.9);
//breaks when no car in leftmost and car in middle lane at the very beginning
const traffic = [];
for (let i = 1; i < numTrafficObstacles * 2; i += 2) {
  let car = new Car(
    road.getLaneCenter(getRandomInt(0, 2)),
    -(i * 100),
    30,
    50,
    "DUMMY"
  );
  let car1 = new Car(
    road.getLaneCenter(getRandomInt(0, 2)),
    -(i * 100),
    30,
    50,
    "DUMMY"
  );
  traffic.push(car);
  traffic.push(car1);
}

const playerCanvas = document.getElementById("playerCanvas");
playerCanvas.width = 200;
const playerCtx = playerCanvas.getContext("2d");
const playerRoad = new Road(playerCanvas.width / 2, playerCanvas.width * 0.9);
const playerCar = new Car(playerRoad.getLaneCenter(1), 100, 30, 50, "KEYS");
const gameTraffic = [];
for (let i = 1; i < numTrafficObstacles * 2; i += 2) {
  let car2 = new Car(
    playerRoad.getLaneCenter(getRandomInt(0, 2)),
    -(i * 100),
    30,
    50,
    "DUMMY"
  );
  let car3 = new Car(
    playerRoad.getLaneCenter(getRandomInt(0, 2)),
    -(i * 100),
    30,
    50,
    "DUMMY"
  );
  gameTraffic.push(car2);
  gameTraffic.push(car3);
}

// Increase N and uncomment save / discard buttons in index.html to train new models
const N = 1;
const cars = generateCars(N);
let bestCar = cars[0];

if (localStorage.getItem("bestBrain")) {
  for (let i = 0; i < cars.length; i++) {
    cars[i].brain = JSON.parse(localStorage.getItem("bestBrain"));

    if (i != 0) {
      NeuralNetwork.mutate(cars[i].brain, 0.25);
    }
  }
} else {
  for (let i = 0; i < cars.length; i++) {
    fetch("src/models/bestBrain.json")
      .then((response) => response.json()) // Parse the JSON response
      .then((data) => {
        cars[i].brain = data; // Assign the parsed data to the brain property
      })
      .catch((error) => console.error("Error loading brain data:", error));
  }
}
var setup2 = false;
var setup = false;
animate();
animatePlayer();

function save() {
  localStorage.setItem("bestBrain", JSON.stringify(bestCar.brain));
}

function discard() {
  localStorage.removeItem("bestBrain");
}

function refresh() {
  location.reload();
}

function generateCars(N) {
  const cars = [];
  for (let i = 1; i <= N; i++) {
    cars.push(new Car(road.getLaneCenter(1), 100, 30, 50, "AI"));
  }
  return cars;
}

function isGameOver() {
  networkCtx.fillStyle = "white";
  networkCtx.font = "30px Arial";
  let p1HasFinished = playerCar.y < gameTraffic[gameTraffic.length - 1].y - 50;
  let AIHasFinished = bestCar.y < traffic[traffic.length - 1].y - 50;

  if (!twoPlayers) {
    // check if either car is damaged
    if (playerCar.damaged) {
      return "YOU LOSE!";
    }
    if (bestCar.damaged) {
      return "YOU WON!";
    }

    // neither car is damaged, so check if either has finished
    if (p1HasFinished) {
      return "YOU WON!";
    }
    if (AIHasFinished) {
      return "YOU LOSE!";
    }

    // neither car is damaged nor has either finished, so continue
    return null;
  } else {
    // 2 players and computer
    let p2HasFinished =
      player2Car.y < player2traffic[player2traffic.length - 1].y - 50;

    // AI car is damaged but we want the players to continue on their own
    if (bestCar.damaged) {
      // neither player is damaged, check if either has finished else and continue if not
      if (!playerCar.damaged && !player2Car.damaged) {
        return p1HasFinished
          ? "PLAYER 1 WINS!"
          : p2HasFinished
          ? "PLAYER 2 WINS!"
          : null;
      }

      // one player must be damaged
      return playerCar.damaged ? "PLAYER 2 WINS!" : "PLAYER 1 WINS!";
    } else if (playerCar.damaged) {
      // p2 and AI are both alive, check if either has finished and continue if not
      if (!player2Car.damaged && !bestCar.damaged) {
        return p2HasFinished
          ? "PLAYER 2 WINS!"
          : AIHasFinished
          ? "COMPUTER WINS!"
          : null;
      }

      // p2 or AI must be damaged
      return player2Car.damaged ? "COMPUTER WINS!" : "PLAYER 2 WINS!";
    } else if (player2Car.damaged) {
      // p1 and AI are both alive, check if either has finished and continue if not
      if (!playerCar.damaged && !bestCar.damaged) {
        return p1HasFinished
          ? "PLAYER 1 WINS!"
          : AIHasFinished
          ? "COMPUTER WINS!"
          : null;
      }

      // p1 or AI must be damaged
      return playerCar.damaged ? "COMPUTER WINS!" : "PLAYER 1 WINS!";
    } else {
      // no cars are damaged, just check if any have finished and continue if not
      return p1HasFinished
        ? "PLAYER 1 WINS!"
        : p2HasFinished
        ? "PLAYER 2 WINS!"
        : AIHasFinished
        ? "COMPUTER WINS!"
        : null;
    }
  }
}

function animatePlayer() {
  let ready = JSON.parse(localStorage.getItem("readyToStartGame"));

  if (!setup2 || ready) {
    playerCanvas.height = window.innerHeight;

    for (let i = 0; i < gameTraffic.length; i++) {
      gameTraffic[i].update(playerRoad.borders, []);
    }

    playerCtx.save();
    playerCtx.translate(0, -playerCar.y + playerCanvas.height * 0.7);

    playerRoad.draw(playerCtx);
    for (let i = 0; i < gameTraffic.length; i++) {
      gameTraffic[i].draw(playerCtx, "yellow");
    }

    playerCar.update(playerRoad.borders, gameTraffic);
    playerCar.draw(playerCtx, "white");

    playerCtx.restore();
  }

  if (!setup2) {
    setup2 = true;
  }

  if (isGameOver() == null) {
    requestAnimationFrame(animatePlayer);
  } else {
    networkCtx.fillText(isGameOver(), 150, 100);
  }
}

function animate(time) {
  let ready = JSON.parse(localStorage.getItem("readyToStartGame"));

  if (!setup || ready) {
    for (let i = 0; i < traffic.length; i++) {
      traffic[i].update(road.borders, []);
    }
    for (let i = 0; i < cars.length; i++) {
      cars[i].update(road.borders, traffic);
    }

    bestCar = cars.find((c) => c.y == Math.min(...cars.map((c) => c.y)));

    carCanvas.height = window.innerHeight;
    networkCanvas.height = window.innerHeight;

    carCtx.save();
    carCtx.translate(0, -bestCar.y + carCanvas.height * 0.7);

    road.draw(carCtx);
    for (let i = 0; i < traffic.length; i++) {
      traffic[i].draw(carCtx, "yellow");
    }

    carCtx.globalAlpha = 0.2;
    for (let i = 0; i < cars.length; i++) {
      cars[i].draw(carCtx, "white");
    }
    carCtx.globalAlpha = 1;
    bestCar.draw(carCtx, "white", true);

    carCtx.restore();
  }
  // mark setup as done
  if (!setup) {
    setup = true;
  }

  if (isGameOver() == null) {
    requestAnimationFrame(animate);
  } else {
    networkCtx.fillText(isGameOver(), 150, 100);
  }
}
