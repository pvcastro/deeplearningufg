Highcharts.chart('container', {
    chart: {
        type: 'bar'
    },
    exporting: {
    		sourceWidth: 1400,
        sourceHeight: 1000
    },
    title: {
        text: ''
    },
    xAxis: {
        categories: ['TJPR', 'TJSP', 'TJRJ', 'TJRS', 'TJMG', 'TJSC', 'TJBA', 'TJGO', 'TJPE', 'TJMT', 'TJDFT', 'TJMA', 'TJPA', 'TJCE', 'TJES', 'TJTO', 'TJMS', 'TJAM', 'TJAL', 'TJAC', 'TJSE', 'TJRR', 'TJAP', 'TJPB', 'TJPI', 'TJRO', 'TJRN','TODOS'],
        title: {
            text: null
        },
        labels: {
        	step: 1,
          style: {
          		color: "#000000"
          }
        }
    },
    yAxis: {
        title: {
            text: '',
            align: 'high'
        },
        labels: {
            overflow: 'justify',
            formatter: function() {
               return Math.abs(this.value) + '%';
            },
            style: {
            	color: "#000000"
            }
        },
        max: 100
    },
    tooltip: {
        valueSuffix: ' %'
    },
    plotOptions: {
        bar: {
            dataLabels: {
                enabled: true,
                formatter: function() {
                   return this.y + '%';
                }
            }
        }
    },
    series: [{
        name: 'Percentual de casos novos eletrônicos',
        data: [99.4,97.6,92.4,39.8,39.5,97.6,93.2,86.3,82.9,82.3,71.6,69.1,65.6,64.3,33.9,100,100,100,100,100,100,96.7,90.3,88.5,86.1,80.3,79.5,82.6]
    }]
});