class Road{
    constructor(x, width, laneCount = 3){
        this.x=x;
        this.width = width; //const road = new Road(canvas.width/2, canvas.width*0.9);
        this.laneCount = laneCount;

        this.left = x-width/2; //10
        this.right = x+width/2; //190

        const infinity = 10000000;
        this.top = -infinity;
        this.bottom = infinity;

        const topLeft = {x:this.left,y:this.top};
        const topRight = {x:this.right,y:this.top};
        const bottomLeft = {x:this.left,y:this.bottom};
        const bottomRight = {x:this.right,y:this.bottom};

        // this.borders = [
        //     [topLeft,bottomLeft],
        //     [topRight,bottomRight]
        // ];
        
        this.borders = [[
            {x:this.right+2.5,y:this.top},
            {x:this.right-2.5,y:this.top},
            {x:this.right-2.5,y:this.bottom},
            {x:this.right+2.5,y:this.bottom}
        ],
        [
            {x:this.left-2.5,y:this.top},
            {x:this.left+2.5,y:this.top},
            {x:this.left+2.5,y:this.bottom},
            {x:this.left-2.5,y:this.bottom}
        ]
        ];
    }

    getLaneCenter(laneIndex){
        const laneWidth = this.width/this.laneCount;
        return this.left + laneWidth / 2 + Math.min(laneIndex,this.laneCount)*laneWidth;
    }

    draw(ctx){
        ctx.lineWidth = 5;
        ctx.strokeStyle = "white";
        for(let i = 1; i<this.laneCount;i++){

            const x = lerp(
                this.left,
                this.right,
                i/this.laneCount
            );
            ctx.setLineDash([20,20]);
            ctx.beginPath();
            ctx.moveTo(x, this.top);
            ctx.lineTo(x, this.bottom);
            ctx.stroke();
        }

        ctx.setLineDash([]);
        // this.borders.forEach(border=>{
        //     ctx.beginPath();
        //     ctx.moveTo(border[0].x, border[0].y);
        //     ctx.lineTo(border[1].x,border[1].y);
        //     ctx.stroke();
        // });
        
        ctx.fillStyle = 'white';
        for(let i = 0; i<this.borders.length;i++){
            ctx.beginPath();
            ctx.moveTo(this.borders[i][0].x,this.borders[i][0].y);
            for(let j = 1; j<this.borders[0].length;j++){
                ctx.lineTo(this.borders[i][j].x, this.borders[i][j].y)
                ctx.fill();
            }
        }
    }
}