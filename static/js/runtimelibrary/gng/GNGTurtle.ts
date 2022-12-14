import { Module, ModuleStore } from "../../compiler/parser/Module.js";
import { Klass, Visibility } from "../../compiler/types/Class.js";
import { booleanPrimitiveType, charPrimitiveType, doublePrimitiveType, intPrimitiveType, stringPrimitiveType, voidPrimitiveType } from "../../compiler/types/PrimitiveTypes.js";
import { Attribute, Method, Parameterlist, Value } from "../../compiler/types/Types.js";
import { RuntimeObject } from "../../interpreter/RuntimeObject.js";
import { Interpreter } from "../../interpreter/Interpreter.js";
import { RectangleHelper } from "../graphics/Rectangle.js";
import { TurtleHelper } from "../graphics/Turtle.js";
import { FilledShapeHelper } from "../graphics/FilledShape.js";
import { ShapeHelper } from "../graphics/Shape.js";
import { GNGFarben } from "./GNGFarben.js";
import { GNGEreignisbehandlung, GNGEreignisbehandlungHelper } from "./GNGEreignisbehandlung.js";

export class GNGTurtleClass extends Klass {

    constructor(module: Module, moduleStore: ModuleStore) {

        let objectType = moduleStore.getType("Object").type;

        super("GTurtle", module, "Turtle-Klasse der Graphics'n Games-Bibliothek (Cornelsen-Verlag)");

        this.addAttribute(new Attribute("x", intPrimitiveType, (value: Value) => { 
            let sh = value.object.intrinsicData["Actor"];
            value.value = Math.round(sh.lineElements[sh.lineElements.length - 1].x); 
        }, false, Visibility.private, false, "x-Position der Figur"));
        this.addAttribute(new Attribute("y", intPrimitiveType, (value: Value) => { 
            let sh = value.object.intrinsicData["Actor"];
            value.value = Math.round(sh.lineElements[sh.lineElements.length - 1].y); 
        }, false, Visibility.private, false, "x-Position der Figur"));

        this.addAttribute(new Attribute("winkel", intPrimitiveType, (value: Value) => { 
            value.value = value.object.intrinsicData["Actor"].angle 
        }, false, Visibility.private, false, "Blickrichtung der Figur in Grad"));

        this.addAttribute(new Attribute("gr????e", intPrimitiveType, (value: Value) => { 
            value.value = Math.round(value.object.intrinsicData["Actor"].scaleFactor*100) 
        }, false, Visibility.private, false, "Gr????e der Figur (100 entspricht 'normalgro??')"));

        this.addAttribute(new Attribute("sichtbar", booleanPrimitiveType, (value: Value) => { 
            value.value = value.object.intrinsicData["Actor"].displayObject?.visible 
        }, false, Visibility.private, false, "true, wenn die Figur sichtbar ist"));

        this.addAttribute(new Attribute("stiftUnten", booleanPrimitiveType, (value: Value) => { 
            value.value = value.object.intrinsicData["Actor"].penIsDown; 
        }, false, Visibility.private, false, "true, wenn die Turtle beim Gehen zeichnet"));

        this.setupAttributeIndicesRecursive();

        this.addMethod(new Method("GTurtle", new Parameterlist([]), null,
            (parameters) => {

                let o: RuntimeObject = parameters[0].value;
                o.intrinsicData["isGNG"] = true;

                let rh = new TurtleHelper(100, 200, true, module.main.getInterpreter(), o);
                rh.setShowTurtle(true);
                rh.setBorderColor(0);
                o.intrinsicData["Actor"] = rh;

                let helper: GNGEreignisbehandlungHelper = GNGEreignisbehandlung.getHelper(module);
                helper.registerEvents(o);

            }, false, false, 'Instanziert ein neues Turtle-Objekt.', true));

        this.addMethod(new Method("gr????eSetzen", new Parameterlist([
            { identifier: "gr????e", type: intPrimitiveType, declaration: null, usagePositions: null, isFinal: true }
        ]), null,
            (parameters) => {

                let o: RuntimeObject = parameters[0].value;
                let sh: TurtleHelper = o.intrinsicData["Actor"];
                let gr????e: number = parameters[1].value;

                if (sh.testdestroyed("gr????eSetzen")) return;

                sh.turtleSize = gr????e;
                sh.turn(0);

            }, false, false, "Setzt die Gr????e des Turtle-Dreiecks.", false));

        this.addMethod(new Method("FarbeSetzen", new Parameterlist([
            { identifier: "farbe", type: stringPrimitiveType, declaration: null, usagePositions: null, isFinal: true },
        ]), null,
            (parameters) => {

                let o: RuntimeObject = parameters[0].value;
                let sh: TurtleHelper = o.intrinsicData["Actor"];
                let farbe: string = parameters[1].value;

                let color: number = GNGFarben[farbe.toLocaleLowerCase()];
                if (color == null) color = 0x000000; // default: schwarz

                if (sh.testdestroyed("FarbeSetzen")) return;

                sh.setBorderColor(color);
                sh.render();

            }, false, false, "Setzt die Zeichenfarbe der Turtle.", false));


        this.addMethod(new Method("Drehen", new Parameterlist([
            { identifier: "grad", type: intPrimitiveType, declaration: null, usagePositions: null, isFinal: true },
        ]), null,
            (parameters) => {

                let o: RuntimeObject = parameters[0].value;
                let sh: TurtleHelper = o.intrinsicData["Actor"];
                let grad: number = parameters[1].value;

                if (sh.testdestroyed("Drehen")) return;

                sh.turn(grad);

            }, false, false, "Dreht die Turtle um den angegebenen Winkel. Positiver Winkel bedeutet Drehung gegen den Uhrzeigersinn.", false));

        this.addMethod(new Method("Gehen", new Parameterlist([
            { identifier: "l??nge", type: doublePrimitiveType, declaration: null, usagePositions: null, isFinal: true },
        ]), null,
            (parameters) => {

                let o: RuntimeObject = parameters[0].value;
                let sh: TurtleHelper = o.intrinsicData["Actor"];
                let l??nge: number = parameters[1].value;

                if (sh.testdestroyed("Gehen")) return;

                sh.forward(l??nge);

            }, false, false, "Bewirkt, dass die Turtle um die angegebene L??nge nach vorw??rts geht.", false));

        this.addMethod(new Method("StiftHeben", new Parameterlist([
        ]), null,
            (parameters) => {

                let o: RuntimeObject = parameters[0].value;
                let sh: TurtleHelper = o.intrinsicData["Actor"];

                if (sh.testdestroyed("StiftHeben")) return;

                sh.penIsDown = false;

            }, false, false, "Bewirkt, dass die Turtle beim Gehen ab jetzt nicht mehr zeichnet.", false));

        this.addMethod(new Method("StiftSenken", new Parameterlist([
        ]), null,
            (parameters) => {

                let o: RuntimeObject = parameters[0].value;
                let sh: TurtleHelper = o.intrinsicData["Actor"];

                if (sh.testdestroyed("StiftSenken")) return;

                sh.penIsDown = false;

            }, false, false, "Bewirkt, dass die Turtle beim Gehen ab jetzt wieder zeichnet.", false));

        this.addMethod(new Method("L??schen", new Parameterlist([
        ]), null,
            (parameters) => {

                let o: RuntimeObject = parameters[0].value;
                let sh: TurtleHelper = o.intrinsicData["Actor"];

                if (sh.testdestroyed("L??schen")) return;

                sh.clear();

            }, false, false, "L??scht alles von der Turtle gezeichnete und versetzt die Turtle in den Ausgangszustand.", false));

        this.addMethod(new Method("PositionSetzen", new Parameterlist([
            { identifier: "x", type: intPrimitiveType, declaration: null, usagePositions: null, isFinal: true },
            { identifier: "y", type: intPrimitiveType, declaration: null, usagePositions: null, isFinal: true }
        ]), null,
            (parameters) => {

                let o: RuntimeObject = parameters[0].value;
                let sh: TurtleHelper = o.intrinsicData["Actor"];
                let x: number = parameters[1].value;
                let y: number = parameters[2].value;

                if (sh.testdestroyed("PositionSetzen")) return;

                sh.moveTo(x, y);

            }, false, false, "Verschiebt die Turtle an die Position (x, y) ohne eine neue Linie zu zeichnen.", false));

        this.addMethod(new Method("ZumStartpunktGehen", new Parameterlist([
        ]), null,
            (parameters) => {

                let o: RuntimeObject = parameters[0].value;
                let sh: TurtleHelper = o.intrinsicData["Actor"];

                if (sh.testdestroyed("ZumStartpunktGehen")) return;

                sh.moveTo(100, 200);

            }, false, false, "Verschiebt die Turtle an die Position (100, 200) ohne eine neue Linie zu zeichnen.", false));

        this.addMethod(new Method("WinkelSetzen", new Parameterlist([
            { identifier: "winkel", type: intPrimitiveType, declaration: null, usagePositions: null, isFinal: true },
        ]), null,
            (parameters) => {

                let o: RuntimeObject = parameters[0].value;
                let sh: TurtleHelper = o.intrinsicData["Actor"];
                let winkel: number = parameters[1].value;

                if (sh.testdestroyed("WinkelSetzen")) return;

                sh.turn(winkel - sh.angle);

            }, false, false, "Setzt den Blickwinkel der Turtle. 0?? => nach rechts, 90??: => nach oben, usw..", false));

        this.addMethod(new Method("WinkelGeben", new Parameterlist([
        ]), intPrimitiveType,
            (parameters) => {

                let o: RuntimeObject = parameters[0].value;
                let sh: TurtleHelper = o.intrinsicData["Actor"];

                if (sh.testdestroyed("WinkelGeben")) return;

                return Math.round(sh.angle);

            }, false, false, "Gibt den Blickwinkel der Turtle zur??ck.", false));

        this.addMethod(new Method("XPositionGeben", new Parameterlist([
        ]), intPrimitiveType,
            (parameters) => {

                let o: RuntimeObject = parameters[0].value;
                let sh: TurtleHelper = o.intrinsicData["Actor"];

                if (sh.testdestroyed("XPositionGeben")) return;

                return Math.round(sh.lineElements[sh.lineElements.length - 1].x);

            }, false, false, "Gibt x-Position der Turtle zur??ck.", false));

        this.addMethod(new Method("YPositionGeben", new Parameterlist([
        ]), intPrimitiveType,
            (parameters) => {

                let o: RuntimeObject = parameters[0].value;
                let sh: TurtleHelper = o.intrinsicData["Actor"];

                if (sh.testdestroyed("YPositionGeben")) return;

                return Math.round(sh.lineElements[sh.lineElements.length - 1].y);

            }, false, false, "Gibt y-Position der Turtle zur??ck.", false));

        this.addMethod(new Method("SichtbarkeitSetzen", new Parameterlist([
            { identifier: "sichtbarkeit", type: intPrimitiveType, declaration: null, usagePositions: null, isFinal: true },
        ]), null,
            (parameters) => {

                let o: RuntimeObject = parameters[0].value;
                let sh: FilledShapeHelper = o.intrinsicData["Actor"];
                let sichtbarkeit: boolean = parameters[1].value;

                if (sh.testdestroyed("SichtbarkeitSetzen")) return;

                sh.setVisible(sichtbarkeit);

            }, false, false, "Schaltet die Sichtbarkeit der Figur ein oder aus.", false));

        this.addMethod(new Method("Entfernen", new Parameterlist([]), null,
            (parameters) => {

                let o: RuntimeObject = parameters[0].value;
                let sh: FilledShapeHelper = o.intrinsicData["Actor"];

                if (sh.testdestroyed("Entfernen")) return;

                sh.destroy();

            }, false, false, "Schaltet die Sichtbarkeit der Figur ein oder aus.", false));

        this.addMethod(new Method("GanzNachVornBringen", new Parameterlist([
        ]), null,
            (parameters) => {

                let o: RuntimeObject = parameters[0].value;
                let sh: FilledShapeHelper = o.intrinsicData["Actor"];

                if (sh.testdestroyed("GanzNachVornBringen")) return;

                return sh.bringToFront();

            }, false, false, 'Setzt das Grafikobjekt vor alle anderen.', false));

        this.addMethod(new Method("GanzNachHintenBringen", new Parameterlist([
        ]), null,
            (parameters) => {

                let o: RuntimeObject = parameters[0].value;
                let sh: FilledShapeHelper = o.intrinsicData["Actor"];

                if (sh.testdestroyed("GanzNachHintenBringen")) return;

                return sh.sendToBack();

            }, false, false, 'Setzt das Grafikobjekt hinter alle anderen.', false));

        this.addMethod(new Method("NachVornBringen", new Parameterlist([
        ]), null,
            (parameters) => {

                let o: RuntimeObject = parameters[0].value;
                let sh: FilledShapeHelper = o.intrinsicData["Actor"];

                if (sh.testdestroyed("NachVornBringen")) return;

                return sh.bringOnePlaneFurtherToFront();

            }, false, false, 'Setzt das Grafikobjekt eine Ebene nach vorne.', false));

        this.addMethod(new Method("NachHintenBringen", new Parameterlist([
        ]), null,
            (parameters) => {

                let o: RuntimeObject = parameters[0].value;
                let sh: FilledShapeHelper = o.intrinsicData["Actor"];

                if (sh.testdestroyed("NachHintenBringen")) return;

                return sh.bringOnePlaneFurtherToBack();

            }, false, false, 'Setzt das Grafikobjekt eine Ebene nach hinten.', false));

        this.addMethod(new Method("Ber??hrt", new Parameterlist([
        ]), booleanPrimitiveType,
            (parameters) => {

                let o: RuntimeObject = parameters[0].value;
                let sh: TurtleHelper = o.intrinsicData["Actor"];

                if (sh.testdestroyed("Ber??hrt")) return;

                return sh.touchesAtLeastOneFigure();

            }, false, false, 'Gibt genau dann true zur??ck, wenn sich an der aktuellen Position der Turtle mindestens eine andere Figur befindet.', false));

        this.addMethod(new Method("Ber??hrt", new Parameterlist([
            { identifier: "farbe", type: stringPrimitiveType, declaration: null, usagePositions: null, isFinal: true }
        ]), booleanPrimitiveType,
            (parameters) => {

                let o: RuntimeObject = parameters[0].value;
                let farbeString: string = parameters[1].value;
                let sh: TurtleHelper = o.intrinsicData["Actor"];

                if (sh.testdestroyed("Ber??hrt")) return;

                let farbe = GNGFarben[farbeString];
                if (farbe == null) farbe = 0;

                return sh.touchesColor(farbe);

            }, false, false, 'Gibt genau dann true zur??ck, wenn sich an der aktuellen Position der Turtle mindestens eine andere Figur mit der angegebenen Farbe befindet.', false));

        this.addMethod(new Method("Ber??hrt", new Parameterlist([
            { identifier: "objekt", type: objectType, declaration: null, usagePositions: null, isFinal: true }
        ]), booleanPrimitiveType,
            (parameters) => {

                let o: RuntimeObject = parameters[0].value;
                let object: RuntimeObject = parameters[1].value;
                let sh: TurtleHelper = o.intrinsicData["Actor"];
                let objectShapeHelper = object.intrinsicData["Actor"];

                if (objectShapeHelper == null || !(objectShapeHelper instanceof ShapeHelper)) return false;

                if (sh.testdestroyed("Ber??hrt")) return;

                return sh.touchesShape(objectShapeHelper);

            }, false, false, 'Gibt genau dann true zur??ck, wenn die ??bergebene Figur die aktuelle Turtleposition enth??lt.', false));



        this.addMethod(new Method("AktionAusf??hren", new Parameterlist([]), voidPrimitiveType,
            null,  // no implementation!
            false, false, "Diese Methode wird vom Taktgeber aufgerufen."));

        this.addMethod(new Method("TasteGedr??ckt", new Parameterlist([
            { identifier: "taste", type: charPrimitiveType, declaration: null, usagePositions: null, isFinal: true }
        ]), voidPrimitiveType,
            null,  // no implementation!
            false, false, "Wird aufgerufen, wenn eine Taste gedr??ckt wird."));

        this.addMethod(new Method("SonderTasteGedr??ckt", new Parameterlist([
            { identifier: "taste", type: intPrimitiveType, declaration: null, usagePositions: null, isFinal: true }
        ]), voidPrimitiveType,
            null,  // no implementation!
            false, false, "Wird aufgerufen, wenn eine Sondertaste gedr??ckt wird."));

        this.addMethod(new Method("MausGecklickt", new Parameterlist([
            { identifier: "x", type: intPrimitiveType, declaration: null, usagePositions: null, isFinal: true },
            { identifier: "y", type: intPrimitiveType, declaration: null, usagePositions: null, isFinal: true },
            { identifier: "anzahl", type: intPrimitiveType, declaration: null, usagePositions: null, isFinal: true },
        ]), voidPrimitiveType,
            null,  // no implementation!
            false, false, "Wird aufgerufen, wenn eine die linke Maustaste gedr??ckt wird."));



    }





}

